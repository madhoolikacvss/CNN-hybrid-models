import torch
from utils.datasets import get_testloader
from models.leafnetv2 import LeafNetv2
from utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PlantVillage test set
test_loader = get_testloader("data/plantvillage", batch_size=64)

# Load trained model
model = LeafNetv2(n_class=len(test_loader.dataset.classes)).to(device)
model.load_state_dict(torch.load("leafnetv2_best.pth"))
model.eval()

# Evaluate
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

compute_metrics(all_labels, all_preds, class_names=test_loader.dataset.classes)
