import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ===========================
# CONFIG
# ===========================
DATA_SPLIT_DIR = "data/split"
EMBEDDINGS_DIR = "data/embeddings"
CHECKPOINT_PATH = "models/simclr_encoder.pth"

BATCH_SIZE = 64
IMAGE_SIZE = 224

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


# ===========================
# TRANSFORM (MATCH SIMCLR)
# ===========================
def get_base_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


# ===========================
# BUILD ENCODER (MUST MATCH TRAINING EXACTLY)
# ===========================
def build_encoder():

    model = models.mobilenet_v2(weights=None)

    encoder = nn.Sequential(
        model.features,

        nn.Conv2d(1280, 1280, kernel_size=1),
        nn.BatchNorm2d(1280),
        nn.ReLU(inplace=True),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )

    return encoder


# ===========================
# LOAD ENCODER WEIGHTS
# ===========================
def load_encoder(device):

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    encoder = build_encoder()
    encoder.load_state_dict(checkpoint["encoder"])

    encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


# ===========================
# DATALOADER
# ===========================
def get_loader(split):
    path = os.path.join(DATA_SPLIT_DIR, split)

    dataset = datasets.ImageFolder(
        root=path,
        transform=get_base_transform()
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return loader, dataset.class_to_idx


# ===========================
# MAIN
# ===========================
def generate_embeddings():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder = load_encoder(device)

    class_map_saved = False

    for split in ["train", "val", "test"]:

        print(f"\nProcessing {split}...")

        loader, class_to_idx = get_loader(split)

        if not class_map_saved:
            with open(os.path.join(EMBEDDINGS_DIR, "class_to_idx.json"), "w") as f:
                json.dump(class_to_idx, f, indent=4)
            class_map_saved = True

        embeddings_list = []
        labels_list = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device)

                outputs = encoder(images)

                outputs = torch.nn.functional.normalize(outputs, dim=1)

                embeddings_list.append(outputs.cpu())
                labels_list.append(labels)

        embeddings = torch.cat(embeddings_list)
        labels = torch.cat(labels_list)

        save_path = os.path.join(EMBEDDINGS_DIR, f"{split}_embeddings.pt")

        torch.save({
            "embeddings": embeddings,
            "labels": labels
        }, save_path)

        print(f"Saved: {save_path}")
        print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    generate_embeddings()