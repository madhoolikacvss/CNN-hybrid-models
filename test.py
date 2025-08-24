# test.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from thop import profile

from utils.datasets import get_plantvillage_loader
from models.leafnetv2 import LeafNetv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE = 64
NUM_CLASSES = 14  # adjust if needed

test_loader = get_plantvillage_loader(batch_size=BATCH_SIZE)

model = LeafNetv2(n_class=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("leafnetv2_best.pth", map_location=device))
model.eval()


# Efficiency metrics on single image

# Take one image from test_loader
inputs, _ = next(iter(test_loader))
single_input = inputs[:1].to(device)  # only first image

# FLOPs and params
flops, params = profile(model, inputs=(single_input,), verbose=False)
print(f"Total parameters: {params/1e6:.2f}M")
print(f"Total FLOPs: {flops/1e9:.2f}G")

# Latency measurement
def measure_latency(model, input_tensor, n=100):
    model.eval()
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    torch.cuda.synchronize() if input_tensor.device.type == "cuda" else None
    start = time.time()
    for _ in range(n):
        _ = model(input_tensor)
    torch.cuda.synchronize() if input_tensor.device.type == "cuda" else None
    avg_time = (time.time() - start)/n
    return avg_time

cpu_time = measure_latency(model, single_input.cpu())
print(f"Avg single-image inference latency on CPU: {cpu_time*1000:.2f} ms")

if device.type == "cuda":
    gpu_time = measure_latency(model, single_input)
    print(f"Avg single-image inference latency on GPU: {gpu_time*1000:.2f} ms")


# Full dataset evaluation

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Classification metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes, digits=3))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, fmt='d', xticklabels=test_loader.dataset.classes,
            yticklabels=test_loader.dataset.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("PlantVillage Confusion Matrix")
plt.show()
