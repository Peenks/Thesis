import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
EMBED_DIM = 128
TEMPERATURE = 0.2

DATA_DIR = "data/augmented"
SAVE_PATH = "models/simclr_encoder.pth"
CHECKPOINT_PATH = "models/checkpoints/last.pth"
OLD_CKPT_PATH = "models/checkpoints/epoch_2.pth"

SAVE_EVERY = 200

os.makedirs("models/checkpoints", exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("RUNNING FIXED SIMCLR VERSION")
print("Using device:", device)


simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


def get_encoder():
    model = models.mobilenet_v2(weights=None)
    encoder = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    return encoder, 1280


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature):
    n = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T)

    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)

    sim = sim / temperature

    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels])

    return F.cross_entropy(sim, labels)


def train_simclr():
    print("Preparing dataset...")

    dataset = SimCLRDataset(DATA_DIR, simclr_transform)
    print(f"Total images: {len(dataset)}")

    print("Testing first image load...")
    test_x1, test_x2 = dataset[0]
    print("First image loaded:", test_x1.shape, test_x2.shape)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print(f"Total batches per epoch: {len(loader)}")

    encoder, dim = get_encoder()
    projector = ProjectionHead(dim, EMBED_DIM)

    encoder = encoder.to(device)
    projector = projector.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=3e-4
    )

    start_epoch = 0
    start_step = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

        encoder.load_state_dict(ckpt["encoder"])
        projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]

        print(f"Resumed at epoch {start_epoch}, step {start_step}")

    elif os.path.exists(OLD_CKPT_PATH):
        print(f"Loading old encoder weights from: {OLD_CKPT_PATH}")
        ckpt = torch.load(OLD_CKPT_PATH, map_location=device)

        encoder.load_state_dict(ckpt["encoder"], strict=False)
        print("Old encoder weights loaded. Projector and optimizer start fresh.")

    else:
        print("No checkpoint found. Training from scratch.")

    print("Starting training...")

    for epoch in range(start_epoch, EPOCHS):
        encoder.train()
        projector.train()

        total_loss = 0.0
        trained_steps = 0

        for step, (x1, x2) in enumerate(loader):
            if epoch == start_epoch and step < start_step:
                continue

            if step == 0:
                print("Got first batch. Sending to device...")

            x1 = x1.to(device)
            x2 = x2.to(device)

            if step == 0:
                print("First batch on device. Running encoder...")

            h1 = encoder(x1)
            h2 = encoder(x2)

            z1 = projector(h1)
            z2 = projector(h2)

            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            trained_steps += 1

            if (step + 1) % 20 == 0:
                print(f"Epoch {epoch + 1} [{step + 1}/{len(loader)}] Loss: {loss.item():.4f}")

            if (step + 1) % SAVE_EVERY == 0:
                torch.save({
                    "epoch": epoch,
                    "step": step + 1,
                    "encoder": encoder.state_dict(),
                    "projector": projector.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, CHECKPOINT_PATH)

                print(f"Saved checkpoint at epoch {epoch}, step {step + 1}")

        avg_loss = total_loss / max(trained_steps, 1)
        print(f"Epoch {epoch + 1} DONE | Avg Loss: {avg_loss:.4f}")

        start_step = 0

        torch.save({
            "epoch": epoch + 1,
            "step": 0,
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict()
        }, CHECKPOINT_PATH)

        print(f"Saved end-of-epoch checkpoint for epoch {epoch + 1}")

    torch.save(encoder, SAVE_PATH)
    print(f"Training complete. Encoder saved to {SAVE_PATH}")


if __name__ == "__main__":
    train_simclr()