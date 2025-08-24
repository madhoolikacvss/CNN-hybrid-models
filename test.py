# test.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.datasets import get_plantvillage_loader
from models.leafnetv2 import LeafNetv2


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Hyperparameters
BATCH_SIZE = 64
NUM_CLASSES = 14  # adjust

# Data loader

test_loader = get_plantvillage_loader(batch_size=BATCH_SIZE)


# Model
model = LeafNetv2(n_class=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("leafnetv2_best.pth", map_location=device))
model.eval()


# Evaluation

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Metrics

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes, digits=3))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, fmt='d', xticklabels=test_loader.dataset.classes,
            yticklabels=test_loader.dataset.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("PlantVillage Confusion Matrix")
plt.show()
