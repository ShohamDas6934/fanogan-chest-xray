import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
CKPT_DIR     = r"D:\fanogan_project\checkpoints"
TEST_NORMAL  = r"D:\fanogan_project\data\test_normal"
TEST_DISEASE = r"D:\fanogan_project\data\test_disease"
RESULTS_DIR  = r"D:\fanogan_project\results"
ANOMALY_DIR  = r"D:\fanogan_project\results\anomaly_maps"
LATENT_DIM   = 128
IMAGE_SIZE   = 128
BATCH_SIZE   = 16
KAPPA        = 1.0
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

print(f"Using device: {DEVICE}")

class XRayDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img), self.paths[idx]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(-1, LATENT_DIM, 1, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Conv2d(512, 1, 4, 1, 0)
    def forward(self, x):
        f = self.features(x)
        return self.head(f), f

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, LATENT_DIM)
        )
    def forward(self, x):
        return self.net(x)

def anomaly_score(x, E, G, D):
    with torch.no_grad():
        z_hat   = E(x)
        x_recon = G(z_hat)

    # full resolution diff
    diff = (x - x_recon) ** 2

    # scale 1 — full 128x128
    s1 = torch.mean(diff, dim=[1,2,3])

    # scale 2 — downsample to 64x64 and score
    x_down     = torch.nn.functional.avg_pool2d(x,     2)
    recon_down = torch.nn.functional.avg_pool2d(x_recon, 2)
    s2 = torch.mean((x_down - recon_down) ** 2, dim=[1,2,3])

    # scale 3 — downsample to 32x32
    x_down2     = torch.nn.functional.avg_pool2d(x_down,     2)
    recon_down2 = torch.nn.functional.avg_pool2d(recon_down, 2)
    s3 = torch.mean((x_down2 - recon_down2) ** 2, dim=[1,2,3])

    # weighted sum — finer scales catch texture, coarser catch structure
    score = s1 + 0.5 * s2 + 0.25 * s3
    return score.cpu().numpy(), x_recon

def save_heatmaps(loader, E, G, D, label, n=5):
    saved = 0
    for imgs, paths in loader:
        if saved >= n:
            break
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            z_hat   = E(imgs)
            x_recon = G(z_hat)
        for i in range(min(len(imgs), n - saved)):
            orig  = imgs[i].squeeze().cpu().numpy()
            recon = x_recon[i].squeeze().cpu().numpy()
            diff  = np.abs(orig - recon)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(orig,  cmap='gray'); axes[0].set_title('Input');          axes[0].axis('off')
            axes[1].imshow(recon, cmap='gray'); axes[1].set_title('Reconstruction'); axes[1].axis('off')
            axes[2].imshow(diff,  cmap='hot');  axes[2].set_title('Anomaly map');    axes[2].axis('off')
            plt.suptitle(label)
            plt.tight_layout()
            plt.savefig(os.path.join(ANOMALY_DIR, f"{label}_{saved+1}.png"), dpi=100)
            plt.close()
            saved += 1
    print(f"  Saved {saved} heatmaps for {label}")

if __name__ == '__main__':
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    E = Encoder().to(DEVICE)
    G.load_state_dict(torch.load(f"{CKPT_DIR}/G_final.pth",
                                 map_location=DEVICE, weights_only=True))
    D.load_state_dict(torch.load(f"{CKPT_DIR}/D_final.pth",
                                 map_location=DEVICE, weights_only=True))
    E.load_state_dict(torch.load(f"{CKPT_DIR}/E2_final.pth",
                                 map_location=DEVICE, weights_only=True))
    G.eval(); D.eval(); E.eval()
    print("All models loaded.")

    print("\nScoring normal images...")
    normal_loader = DataLoader(XRayDataset(TEST_NORMAL, transform),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    normal_scores = []
    for imgs, _ in tqdm(normal_loader):
        scores, _ = anomaly_score(imgs.to(DEVICE), E, G, D)
        normal_scores.extend(scores.tolist())

    print("Scoring disease images...")
    disease_loader = DataLoader(XRayDataset(TEST_DISEASE, transform),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    disease_scores = []
    for imgs, _ in tqdm(disease_loader):
        scores, _ = anomaly_score(imgs.to(DEVICE), E, G, D)
        disease_scores.extend(scores.tolist())

    all_scores = normal_scores + disease_scores
    all_labels = [0] * len(normal_scores) + [1] * len(disease_scores)
    auroc = roc_auc_score(all_labels, all_scores)

    print(f"\n{'='*40}")
    print(f"  Normal  — mean score : {np.mean(normal_scores):.4f}")
    print(f"  Disease — mean score : {np.mean(disease_scores):.4f}")
    print(f"  AUROC                : {auroc:.4f}")
    print(f"{'='*40}")

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0,1],[0,1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("f-AnoGAN ROC Curve — ChestX-ray14")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("ROC curve saved to results/roc_curve.png")

    print("\nGenerating anomaly heatmaps...")
    save_heatmaps(normal_loader,  E, G, D, "normal",  n=5)
    save_heatmaps(disease_loader, E, G, D, "disease", n=5)
    print("\nDone. Check D:\\fanogan_project\\results\\")