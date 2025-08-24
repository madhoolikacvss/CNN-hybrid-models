# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
from thop import profile  # for FLOPs
from utils.datasets import get_newplant_loaders
from models.leafnetv2 import LeafNetv2
from utils.train_utils import train_one_epoch, evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 30
NUM_CLASSES = 14  # adjust if needed


# Data loaders

train_loader, val_loader = get_newplant_loaders(batch_size=BATCH_SIZE)


# Model
model = LeafNetv2(n_class=NUM_CLASSES).to(device)

# Count params and FLOPs
dummy_input = torch.randn(1, 3, 224, 224).to(device)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
print(f"Total params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.2f}G")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# Training loop
best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    start_time = time.time()
    
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}/{NUM_EPOCHS} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f}s")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "leafnetv2_best.pth")
        print(f"Saved best model at epoch {epoch} with val_acc {val_acc:.4f}")
