import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from train_finetune import FineTunedTrashClassifier

MODEL_PATH = "models/finetuned_model.pth"
DATA_DIR = "data/split/test"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Recreate encoder
mobilenet = models.mobilenet_v2(weights=None)

encoder = nn.Sequential(
    mobilenet.features,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

model = FineTunedTrashClassifier(encoder, NUM_CLASSES)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = []
y_pred = []

with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(device)

        outputs = model(xb)
        preds = outputs.argmax(dim=1)

        y_true.extend(yb.numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

# Normalize row-wise (each row sums to 100%)
cm = cm.astype("float") / cm.sum(axis=1)[:, None]
cm = cm * 100

class_names = dataset.classes

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Normalized Confusion Matrix (%)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)

print("Saved as confusion_matrix.png")
plt.show()