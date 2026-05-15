import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
TRAIN_DIR  = r"D:\fanogan_project\data\train_normal"
CKPT_DIR   = r"D:\fanogan_project\checkpoints"
LATENT_DIM = 128
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS     = 100
LR         = 1e-4
LAMBDA_GP  = 10
N_CRITIC   = 5
SAVE_EVERY = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
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

dataset    = XRayDataset(TRAIN_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True)
print(f"Dataset size: {len(dataset)} images")

# ── Generator (128x128 output) ────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # z: (128,) → (512, 4, 4)
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            # → (256, 8, 8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            # → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            # → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            # → (32, 64, 64)
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            # → (1, 128, 128)
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(-1, LATENT_DIM, 1, 1))

# ── Discriminator (128x128 input) ─────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # (1, 128, 128) → (32, 64, 64)
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            # → (64, 32, 32)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            # → (128, 16, 16)
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            # → (256, 8, 8)
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            # → (512, 4, 4)
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, x):
        f = self.features(x)
        return self.head(f), f

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.9))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))

# ── gradient penalty ──────────────────────────────────────────────────────────
def gradient_penalty(D, real, fake):
    alpha  = torch.rand(real.size(0), 1, 1, 1, device=DEVICE)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out, _ = D(interp)
    grads  = torch.autograd.grad(d_out, interp,
                                 grad_outputs=torch.ones_like(d_out),
                                 create_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

# ── training loop ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    start_epoch = 1
    latest_ckpt = 0
    for f in os.listdir(CKPT_DIR):
        if f.startswith("G_epoch") and f.endswith(".pth"):
            num = int(f.replace("G_epoch", "").replace(".pth", ""))
            if num > latest_ckpt:
                latest_ckpt = num

    if latest_ckpt > 0:
        G.load_state_dict(torch.load(f"{CKPT_DIR}/G_epoch{latest_ckpt}.pth",
                                     map_location=DEVICE, weights_only=True))
        D.load_state_dict(torch.load(f"{CKPT_DIR}/D_epoch{latest_ckpt}.pth",
                                     map_location=DEVICE, weights_only=True))
        start_epoch = latest_ckpt + 1
        print(f"Resumed from epoch {latest_ckpt} — continuing from epoch {start_epoch}")
    else:
        print("No checkpoint found — starting fresh")

    for epoch in range(start_epoch, EPOCHS + 1):
        G.train(); D.train()
        loss_D_total = loss_G_total = 0

        for real_imgs in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
            real_imgs = real_imgs.to(DEVICE)
            b = real_imgs.size(0)

            for _ in range(N_CRITIC):
                z         = torch.randn(b, LATENT_DIM, device=DEVICE)
                fake      = G(z).detach()
                d_real, _ = D(real_imgs)
                d_fake, _ = D(fake)
                gp        = gradient_penalty(D, real_imgs, fake)
                loss_D    = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()
                loss_D_total += loss_D.item()

            z      = torch.randn(b, LATENT_DIM, device=DEVICE)
            fake   = G(z)
            loss_G = -D(fake)[0].mean()
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            loss_G_total += loss_G.item()

        avg_D = loss_D_total / (len(dataloader) * N_CRITIC)
        avg_G = loss_G_total / len(dataloader)
        print(f"Epoch {epoch} | D loss: {avg_D:.4f} | G loss: {avg_G:.4f}")

        if epoch % SAVE_EVERY == 0:
            torch.save(G.state_dict(), f"{CKPT_DIR}/G_epoch{epoch}.pth")
            torch.save(D.state_dict(), f"{CKPT_DIR}/D_epoch{epoch}.pth")
            print(f"  Checkpoints saved at epoch {epoch}")

    torch.save(G.state_dict(), f"{CKPT_DIR}/G_final.pth")
    torch.save(D.state_dict(), f"{CKPT_DIR}/D_final.pth")
    print("Training complete. Final models saved.")