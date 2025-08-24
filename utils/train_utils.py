import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss/n, correct/n

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss/n, correct/n
