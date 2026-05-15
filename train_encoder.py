import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
TRAIN_NORMAL  = r"D:\fanogan_project\data\train_normal"
TEST_DISEASE  = r"D:\fanogan_project\data\test_disease"
CKPT_DIR      = r"D:\fanogan_project\checkpoints"
LATENT_DIM    = 128
IMAGE_SIZE    = 128
BATCH_SIZE    = 16
EPOCHS        = 60
LR            = 1e-4
KAPPA         = 0.1
MARGIN        = 0.05
SAVE_EVERY    = 6
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

print(f"Using device: {DEVICE}")

# ── dataset ──────────────────────────────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

normal_dataset  = XRayDataset(TRAIN_NORMAL, transform)
disease_dataset = XRayDataset(TEST_DISEASE, transform)

normal_loader  = DataLoader(normal_dataset,  batch_size=BATCH_SIZE,
                            shuffle=True,  num_workers=0, pin_memory=True)
disease_loader = DataLoader(disease_dataset, batch_size=BATCH_SIZE,
                            shuffle=True,  num_workers=0, pin_memory=True)

print(f"Normal images : {len(normal_dataset)}")
print(f"Disease images: {len(disease_dataset)}")

# ── models ───────────────────────────────────────────────────────────────────
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

if __name__ == '__main__':
    # ── load frozen G and D ───────────────────────────────────────────────────
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.load_state_dict(torch.load(f"{CKPT_DIR}/G_final.pth",
                                 map_location=DEVICE, weights_only=True))
    D.load_state_dict(torch.load(f"{CKPT_DIR}/D_final.pth",
                                 map_location=DEVICE, weights_only=True))
    for p in G.parameters(): p.requires_grad_(False)
    for p in D.parameters(): p.requires_grad_(False)
    G.eval(); D.eval()
    print("Loaded and froze G and D")

    # ── encoder with resume ───────────────────────────────────────────────────
    E = Encoder().to(DEVICE)
    opt_E = torch.optim.Adam(E.parameters(), lr=LR, betas=(0.5, 0.9))

    start_epoch = 1
    latest_ckpt = 0
    for f in os.listdir(CKPT_DIR):
        if f.startswith("E2_epoch") and f.endswith(".pth"):
            num = int(f.replace("E2_epoch", "").replace(".pth", ""))
            if num > latest_ckpt:
                latest_ckpt = num

    if latest_ckpt > 0:
        E.load_state_dict(torch.load(f"{CKPT_DIR}/E2_epoch{latest_ckpt}.pth",
                                     map_location=DEVICE, weights_only=True))
        start_epoch = latest_ckpt + 1
        print(f"Resumed from epoch {latest_ckpt} — continuing from {start_epoch}")
    else:
        print("No checkpoint found — starting fresh")

    disease_iter = iter(disease_loader)

    for epoch in range(start_epoch, EPOCHS + 1):
        E.train()
        loss_total = 0

        for normal_imgs in tqdm(normal_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            normal_imgs = normal_imgs.to(DEVICE)

            # get a batch of disease images
            try:
                disease_imgs = next(disease_iter)
            except StopIteration:
                disease_iter = iter(disease_loader)
                disease_imgs = next(disease_iter)
            disease_imgs = disease_imgs.to(DEVICE)

            # ── normal reconstruction loss (minimise) ─────────────────────────
            z_norm   = E(normal_imgs)
            x_norm   = G(z_norm)
            loss_normal = torch.mean((normal_imgs - x_norm) ** 2)

            _, f_real  = D(normal_imgs)
            _, f_recon = D(x_norm)
            loss_feat  = torch.mean((f_real - f_recon) ** 2)

            # ── disease margin loss (push recon error above margin) ───────────
            b_d = min(disease_imgs.size(0), normal_imgs.size(0))
            disease_imgs = disease_imgs[:b_d]
            z_dis  = E(disease_imgs)
            x_dis  = G(z_dis)
            err_dis = torch.mean((disease_imgs - x_dis) ** 2, dim=[1,2,3])
            # penalise if disease recon error is too low (below margin)
            margin_loss = torch.mean(torch.clamp(MARGIN - err_dis, min=0))

            loss_E = loss_normal + KAPPA * loss_feat + 2.0 * margin_loss
            opt_E.zero_grad(); loss_E.backward(); opt_E.step()
            loss_total += loss_E.item()

        avg = loss_total / len(normal_loader)
        print(f"Epoch {epoch} | Loss: {avg:.6f}")

        if epoch % SAVE_EVERY == 0:
            torch.save(E.state_dict(), f"{CKPT_DIR}/E2_epoch{epoch}.pth")
            print(f"  Checkpoint saved at epoch {epoch}")

    torch.save(E.state_dict(), f"{CKPT_DIR}/E2_final.pth")
    print("Done. E2_final.pth saved.")