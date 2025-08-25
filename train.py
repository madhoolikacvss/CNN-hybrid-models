# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from collections import Counter

from utils.datasets import get_newplant_loaders
from utils.train_utils import train_one_epoch, evaluate
from utils.metrics import compute_class_weights, compute_flops_params, measure_latency, compute_classification_metrics

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
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--classes", type=int, default=38)
parser.add_argument("--target_acc", type=float, default=0.95)
args = parser.parse_args()

# ---------------------------
# Model selection
# ---------------------------
if args.model == "leafnet":
    model = LeafNetv2(n_class=args.classes)
elif args.model == "efficientnetv2":
    model = EfficientNetV2(num_classes=args.classes)
elif args.model == "mobilenet":
    model = MobileNet(num_classes=args.classes)

model.to(device)
print(f"Training {args.model} on {device}")

# ---------------------------
# Data
# ---------------------------
train_loader, val_loader, _ = get_newplant_loaders(batch_size=args.batch_size)

# ---------------------------
# Class weights
# ---------------------------
weights = compute_class_weights(train_loader, args.classes, device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ---------------------------
# FLOPs & params
# ---------------------------
with torch.no_grad():
    inputs, _ = next(iter(train_loader))
    single_input = inputs[:1].to(device)
    flops, params = compute_flops_params(model, single_input)
    print(f"[{args.model}] Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.2f}G")

# ---------------------------
# Training loop
# ---------------------------
best_val_acc = 0.0
time_to_target = None
training_start_time = time.time()

for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()
    
    # Train & evaluate
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # Compute additional metrics
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    val_acc, val_macro_f1 = compute_classification_metrics(all_labels, all_preds, val_loader.dataset.classes)
    
    elapsed = time.time() - epoch_start
    
    print(f"[{args.model}] Epoch {epoch}/{args.epochs} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Val Macro-F1: {val_macro_f1:.4f} | Time: {elapsed:.1f}s")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{args.model}_best.pth")
        print(f"Saved best {args.model} model (epoch {epoch}, val_acc {val_acc:.4f})")
    
    # Time-to-target
    if time_to_target is None and val_acc >= args.target_acc:
        time_to_target = time.time() - training_start_time
        print(f"{args.model} reached {args.target_acc*100:.1f}% acc in {time_to_target:.1f}s")

# ---------------------------
# Final latency measurement
# ---------------------------
cpu_time = measure_latency(model, single_input.cpu(), device='cpu')
gpu_time = measure_latency(model, single_input.to(device), device='cuda')
print(f"Avg single-image CPU latency: {cpu_time*1000:.2f} ms, GPU latency: {gpu_time*1000:.2f} ms")
