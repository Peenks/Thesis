import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# =====================
# CONFIG
# =====================
DATA_SPLIT_DIR = "data/split"
MODEL_PATH = "models/finetuned_model.pth"
CLASS_MAP_PATH = "data/embeddings/class_to_idx.json"
OUTPUT_IMAGE = "tsne_finetuned_visualization.png"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

USE_PCA_FIRST = True
PCA_DIM = 50

TSNE_PERPLEXITY = 30
TSNE_ITER = 1000


# =====================
# DEVICE
# =====================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =====================
# TRANSFORM
# =====================
def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


# =====================
# ENCODER
# =====================
def build_encoder():
    base = models.mobilenet_v2(weights=None)

    return nn.Sequential(
        base.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )


# =====================
# MODEL
# =====================
class FineTunedTrashClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        return self.encoder(x)


# =====================
# LOAD MODEL
# =====================
def load_model(device, num_classes):
    print("Loading fine-tuned model...")

    encoder = build_encoder()
    model = FineTunedTrashClassifier(encoder, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    return model


# =====================
# EXTRACT FEATURES
# =====================
def extract_finetuned_embeddings(model, loader, device):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)

            features = model.extract_features(images)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            if (batch_idx + 1) % 20 == 0:
                print(f"Processed {(batch_idx + 1) * BATCH_SIZE} images")

    X = torch.cat(all_features).numpy()
    y = torch.cat(all_labels).numpy()

    return X, y


# =====================
# PCA
# =====================
def apply_pca(X):
    print("Applying PCA...")
    pca = PCA(n_components=PCA_DIM)
    return pca.fit_transform(X)


# =====================
# t-SNE
# =====================
def run_tsne(X):
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITER,
        init="pca",
        random_state=42
    )
    return tsne.fit_transform(X)


# =====================
# PLOT
# =====================
def plot(tsne_results, labels):
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.8
    )

    plt.title("t-SNE Visualization of Fine-Tuned Model Embeddings", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Class ID")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.show()

    print(f"Saved visualization to: {OUTPUT_IMAGE}")


# =====================
# MAIN
# =====================
def main():
    device = get_device()
    print(f"Using device: {device}")

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_SPLIT_DIR, "test"),
        transform=get_test_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, "r") as f:
            class_to_idx = json.load(f)
    else:
        class_to_idx = test_dataset.class_to_idx

    num_classes = len(class_to_idx)

    model = load_model(device, num_classes)

    X, y = extract_finetuned_embeddings(model, test_loader, device)

    print(f"Loaded embeddings shape: {X.shape}")
    print(f"Loaded labels shape: {y.shape}")

    if USE_PCA_FIRST:
        X = apply_pca(X)
        print(f"After PCA: {X.shape}")

    tsne_results = run_tsne(X)

    plot(tsne_results, y)


if __name__ == "__main__":
    main()