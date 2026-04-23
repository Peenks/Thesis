import os
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ===========================
# CONFIG
# ===========================
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10              # <--- STOP AT EPOCH 10
EMBED_DIM = 128
TEMPERATURE = 0.5

DATA_DIR = "data/augmented"
SAVE_PATH = "models/simclr_encoder.pth"
CHECKPOINT_DIR = "models/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# ===========================
# DATA TRANSFORMS FOR SIMCLR
# ===========================
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])


# ===========================
# 1. SimCLR Dataset Wrapper
# ===========================
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


# ===========================
# 2. MobileNetV2-inspired Encoder
# ===========================
def get_mobilenetv2_encoder():
    model = models.mobilenet_v2(weights=None)
    encoder = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return encoder, 1280


# ===========================
# 3. Projection Head
# ===========================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


# ===========================
# 4. NT-Xent Loss
# ===========================
def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    reps = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(reps, reps.T)

    mask = torch.eye(batch_size * 2, device=similarity.device).bool()
    similarity = similarity[~mask].view(batch_size * 2, -1)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives], dim=0)

    logits = similarity / temperature
    labels = torch.zeros(batch_size * 2, dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss


# ===========================
# TRAINING LOOP
# ===========================
def train_simclr():

    # dataset
    train_dataset = SimCLRDataset(DATA_DIR, simclr_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # encoder + projector
    encoder, feat_dim = get_mobilenetv2_encoder()
    projector = ProjectionHead(feat_dim, EMBED_DIM)

    encoder = encoder.to(device)
    projector = projector.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=3e-4
    )

    # -------------------------
    # Resume From Checkpoint
    # -------------------------
    checkpoint_files = sorted(glob.glob(f"{CHECKPOINT_DIR}/epoch_*.pth"))

    start_epoch = 0
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"Loading checkpoint: {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]

        print(f"Resuming from epoch {start_epoch}")

    print("Starting SimCLR training...")
    print(f"Total images: {len(train_dataset)}")

    # --------------------
    # TRAIN LOOP
    # --------------------
    for epoch in range(start_epoch, EPOCHS):

        encoder.train()
        projector.train()
        total_loss = 0

        for idx, (x1, x2) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)

            h1 = encoder(x1)
            h2 = encoder(x2)

            z1 = projector(h1)
            z2 = projector(h2)

            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (idx + 1) % 20 == 0:
                print(f"Epoch {epoch+1} [{idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        # -------------------------
        # SAVE CHECKPOINT
        # -------------------------
        ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict()
        }, ckpt_path)

        print(f"Checkpoint saved: {ckpt_path}")

    # -------------------------
    # SAVE FINAL ENCODER ONLY
    # -------------------------
    print("\nTraining complete.")
    torch.save(encoder, SAVE_PATH)
    print(f"Final encoder saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train_simclr()