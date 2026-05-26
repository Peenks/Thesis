import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report


# ===== CONFIG =====
EMBEDDINGS_DIR = "data/embeddings"
MODEL_SAVE_PATH = "models/mlp_classifier.pth"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
PATIENCE = 5


# ===== DEVICE =====
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ===== LOAD EMBEDDINGS =====
def load_split(split_name):
    path = os.path.join(EMBEDDINGS_DIR, f"{split_name}_embeddings.pt")
    data = torch.load(path, map_location="cpu")

    X = data["embeddings"]
    y = data["labels"]

    return X.float(), y.long()


# ===== COMPUTE CLASS WEIGHTS =====
def compute_class_weights(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts)
    return weights


# ===== MODEL =====
class LightweightMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
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
        return self.classifier(x)


# ===== VALIDATION =====
def evaluate(model, loader, device, target_names=None):
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    val_acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    return val_acc, macro_f1, report


# ===== TRAINING LOOP =====
def train():
    device = get_device()
    print(f"Using device: {device}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    input_dim = X_train.shape[1]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    target_names = None
    class_map_path = os.path.join(EMBEDDINGS_DIR, "class_to_idx.json")
    if os.path.exists(class_map_path):
        with open(class_map_path, "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)
    print(f"Class weights: {class_weights}")

    model = LightweightMLP(input_dim, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_macro_f1 = 0.0
    epochs_without_improvement = 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        val_acc, val_macro_f1, val_report = evaluate(
            model,
            val_loader,
            device,
            target_names=target_names
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            print("\n✅ Best model updated and saved.")
            print("=== Validation Classification Report ===")
            print(val_report)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= PATIENCE:
            print("\n⏹ Early stopping triggered.")
            break

    print(f"\n✅ Training finished. Best model saved to {MODEL_SAVE_PATH}")
    print(f"Best Validation Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    train()