import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# =====================
# CONFIG
# =====================
EMBEDDINGS_PATH = "data/embeddings/test_embeddings.pt"
OUTPUT_IMAGE = "tsne_visualization.png"

USE_PCA_FIRST = True   # important for speed + stability
PCA_DIM = 50

TSNE_PERPLEXITY = 30
TSNE_ITER = 1000


# =====================
# LOAD EMBEDDINGS
# =====================
def load_embeddings(path):
    data = torch.load(path, map_location="cpu")
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    return X, y


# =====================
# RUN PCA (optional but recommended)
# =====================
def apply_pca(X):
    print("Applying PCA...")
    pca = PCA(n_components=PCA_DIM)
    return pca.fit_transform(X)


# =====================
# RUN t-SNE
# =====================
def run_tsne(X):
    print("Running t-SNE (this may take a while)...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_ITER,
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

    plt.title("t-SNE Visualization of SimCLR Embeddings", fontsize=14)
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
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_PATH}")

    X, y = load_embeddings(EMBEDDINGS_PATH)

    print(f"Loaded embeddings shape: {X.shape}")
    print(f"Loaded labels shape: {y.shape}")

    # Optional PCA (HIGHLY recommended for SimCLR)
    if USE_PCA_FIRST:
        X = apply_pca(X)
        print(f"After PCA: {X.shape}")

    tsne_results = run_tsne(X)

    plot(tsne_results, y)


if __name__ == "__main__":
    main()