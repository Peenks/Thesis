import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ===========================
# SEED + DEVICE
# ===========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===========================
# CONFIG
# ===========================
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
EMBED_DIM = 128
TEMPERATURE = 0.2   # 🔥 improved (important for better separation)

DATA_DIR = "data/raw/original"

SAVE_PATH = "models/simclr_encoder.pth"
CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ===========================
# SIMCLR AUGMENTATION (IMPROVED)
# ===========================
class SimCLRTransform:
    def __init__(self, size=224):

        self.transform = transforms.Compose([
            # 🔥 stronger SimCLR crop (key improvement)
            transforms.RandomResizedCrop(
                size,
                scale=(0.2, 1.0),
                ratio=(0.75, 1.33)
            ),

            transforms.RandomHorizontalFlip(p=0.5),

            # 🔥 stronger + more stable color jitter
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8,
                    contrast=0.8,
                    saturation=0.8,
                    hue=0.2
                )
            ], p=0.8),

            transforms.RandomGrayscale(p=0.2),

            # 🔥 stronger blur (SimCLR standard improvement)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23)
            ], p=0.5),

            transforms.ToTensor(),

            # 🔥 normalization improves convergence stability
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __call__(self, x):
        return self.transform(x)


simclr_transform = SimCLRTransform(IMAGE_SIZE)


# ===========================
# DATASET (SIMCLR STYLE)
# ===========================
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        # 🔥 two independent views = core SimCLR idea
        return self.transform(img), self.transform(img)


# ===========================
# IMPROVED MOBILENETV2 ENCODER
# ===========================
def get_mobilenetv2_encoder():
    model = models.mobilenet_v2(weights=None)

    backbone = model.features

    # 🔥 KEY IMPROVEMENT:
    # 1x1 conv refines channel interactions BEFORE pooling
    encoder = nn.Sequential(
        backbone,

        nn.Conv2d(1280, 1280, kernel_size=1),
        nn.BatchNorm2d(1280),
        nn.ReLU(inplace=True),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )

    return encoder, 1280


# ===========================
# PROJECTION HEAD (IMPROVED)
# ===========================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),

            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(input_dim // 2, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


# ===========================
# NT-XENT LOSS (STABLE VERSION)
# ===========================
def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z.device)

    return F.cross_entropy(sim, labels)


# ===========================
# TRAINING LOOP
# ===========================
def train_simclr():

    train_dataset = SimCLRDataset(DATA_DIR, simclr_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True   # 🔥 speed improvement
    )

    encoder, feat_dim = get_mobilenetv2_encoder()
    projector = ProjectionHead(feat_dim, EMBED_DIM)

    encoder = encoder.to(device)
    projector = projector.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=3e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    print("Starting SimCLR training...")
    print(f"Total images: {len(train_dataset)}")

    for epoch in range(EPOCHS):

        encoder.train()
        projector.train()

        total_loss = 0

        for idx, (x1, x2) in enumerate(train_loader):

            x1, x2 = x1.to(device), x2.to(device)

            h1 = encoder(x1)
            h2 = encoder(x2)

            z1 = projector(h1)
            z2 = projector(h2)

            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            if (idx + 1) % 20 == 0:
                print(f"Epoch {epoch+1} [{idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # ===========================
        # CHECKPOINT
        # ===========================
        ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict()
        }, ckpt_path)

        print(f"Checkpoint saved: {ckpt_path}")

    print("\nTraining complete.")

    torch.save(encoder.state_dict(), SAVE_PATH)
    print(f"Final encoder saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train_simclr()