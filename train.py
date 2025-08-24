import torch
import torch.nn as nn
import torch.optim as optim
from utils.datasets import get_dataloaders
from utils.train_utils import train_one_epoch, evaluate
from models.leafnetv2 import LeafNetv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_loader, val_loader = get_dataloaders("data/new_plant_diseases", batch_size=64)

# Init model
model = LeafNetv2(n_class=len(train_loader.dataset.dataset.classes)).to(device)

# Loss + optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

# Training 
best_acc = 0
for epoch in range(30):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "leafnetv2_best.pth")
