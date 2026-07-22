import torch
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataset import get_dataloaders
from utils.model_loader import load_model

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMAGE_SIZE = 224
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
_, test_loader, class_names = get_dataloaders(
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    num_workers=0
)

# Model
model = load_model(len(class_names))
model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("="*60)
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()