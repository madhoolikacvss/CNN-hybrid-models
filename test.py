# test.py
import torch
import argparse
from thop import profile
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.datasets import get_newplant_loaders
from utils.metrics import compute_classification_metrics, compute_flops_params, measure_latency

from models.leafnetV2 import LeafNetv2
from models.efficientnetV2 import EfficientNetV2
from models.mobilenet import MobileNet

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
# FLOPs & params
# ---------------------------
with torch.no_grad():
    inputs, _ = next(iter(test_loader))
    single_input = inputs[:1].to(device)
    flops, params = compute_flops_params(model, single_input)
    print(f"[{args.model}] Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.2f}G")

# ---------------------------
# Latency
# ---------------------------
cpu_time = measure_latency(model, single_input.cpu(), device='cpu')
gpu_time = measure_latency(model, single_input.to(device), device='cuda')
print(f"Avg CPU latency: {cpu_time*1000:.2f} ms, GPU latency: {gpu_time*1000:.2f} ms")

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

acc, macro_f1 = compute_classification_metrics(all_labels, all_preds, test_loader.dataset.classes)
print(f"[{args.model}] Test Accuracy:
