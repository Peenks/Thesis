import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report


# =====================
# CONFIG
# =====================
EMBEDDINGS_DIR = "data/embeddings"
MODEL_SAVE_PATH = "models/mlp_classifier.pth"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5


# =====================
# DEVICE
# =====================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================
# LOAD EMBEDDINGS (THIS IS YOUR "TENSOR RESULTS")
# =====================
def load_split(split_name):
    path = os.path.join(EMBEDDINGS_DIR, f"{split_name}_embeddings.pt")
    data = torch.load(path, map_location="cpu")

    embeddings = data["embeddings"].float()   # <-- TENSOR FEATURES (from SimCLR)
    labels = data["labels"].long()            # <-- TENSOR LABELS

    return embeddings, labels


# =====================
# MODEL
# =====================
class LightweightMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = nn.functional.normalize(x, dim=1)
        return self.net(x)


# =====================
# EVALUATION
# =====================
def evaluate(model, loader, device):
    model.eval()

    preds_all = []
    labels_all = []
    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

            preds_all.append(preds.cpu())
            labels_all.append(yb.cpu())

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()

    acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, zero_division=0)

    return acc, macro_f1, report


# =====================
# TRAIN
# =====================
def train():
    device = get_device()
    print("Using device:", device)

    # =====================
    # LOAD SIMCLR EMBEDDINGS (THIS IS THE KEY STEP)
    # =====================
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    X_train, X_val = X_train.float(), X_val.float()

    input_dim = X_train.shape[1]
    num_classes = int(y_train.max().item() + 1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = LightweightMLP(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0
    patience_counter = 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc, val_f1, report = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {total_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            print("✔ Saved best model")
            print(report)

        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    train()