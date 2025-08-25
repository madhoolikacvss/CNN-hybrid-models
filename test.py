# test.py
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from thop import profile
import argparse

# Dataset
from utils.datasets import get_newplant_loaders

# Models
from models.leafnetV2 import LeafNetv2
from models.efficientnetV2 import EfficientNetV2
from models.mobilenet import MobileNet

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="leafnet",
                    choices=["leafnet", "efficientnetv2", "mobilenet"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--classes", type=int, default=38)
args = parser.parse_args()

# ---------------------------
# Load test data
# ---------------------------
_, _, test_loader = get_newplant_loaders(batch_size=args.batch_size)

# ---------------------------
# Load model
# ---------------------------
if args.model=="leafnet":
    model = LeafNetv2(n_class=args.classes)
elif args.model=="efficientnetv2":
    model = EfficientNetV2(num_classes=args.classes)
elif args.model=="mobilenet":
    model = MobileNet(num_classes=args.classes)

checkpoint_path = f"{args.model}_best.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
print(f"Loaded {args.model} checkpoint from {checkpoint_path}")

# ---------------------------
# FLOPs and params
# ---------------------------
with torch.no_grad():
    inputs, _ = next(iter(test_loader))
    single_input = inputs[:1].to(device)
    flops, params = profile(model, inputs=(single_input,), verbose=False)
    print(f"[{args.model}] Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.2f}G")

# ---------------------------
# Latency measurement
# ---------------------------
def measure_latency(model, input_tensor, n=100):
    model.eval()
    for _ in range(10): _ = model(input_tensor)  # warm-up
    torch.cuda.synchronize() if input_tensor.device.type=="cuda" else None
    start = time.time()
    for _ in range(n): _ = model(input_tensor)
    torch.cuda.synchronize() if input_tensor.device.type=="cuda" else None
    return (time.time() - start)/n

cpu_time = measure_latency(model, single_input.cpu())
print(f"[{args.model}] Avg single-image CPU latency: {cpu_time*1000:.2f} ms")
if device.type=="cuda":
    gpu_time = measure_latency(model, single_input)
    print(f"[{args.model}] Avg single-image GPU latency: {gpu_time*1000:.2f} ms")

# ---------------------------
# Evaluation
# ---------------------------
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Metrics
accuracy = sum([p==l for p,l in zip(all_preds, all_labels)]) / len(all_labels)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"[{args.model}] Test Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"{args.model} Confusion Matrix")
plt.show()
