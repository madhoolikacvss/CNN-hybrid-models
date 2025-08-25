import time
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile

# ---------------------------
# Classification metrics
# ---------------------------
def compute_classification_metrics(y_true, y_pred, class_names=None, plot_cm=True):
    """
    Computes accuracy, macro-F1 score, prints classification report, and optionally plots confusion matrix.
    
    Args:
        y_true (list or array): True labels
        y_pred (list or array): Predicted labels
        class_names (list): Optional class names for report and confusion matrix
        plot_cm (bool): Whether to plot confusion matrix
    
    Returns:
        acc (float): Overall accuracy
        macro_f1 (float): Macro-F1 score
    """
    acc = sum([p==t for p,t in zip(y_pred, y_true)]) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    if class_names is None:
        class_names = [str(i) for i in range(len(set(y_true)))]
    
    print(f"Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    
    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    
    return acc, macro_f1

# ---------------------------
# Class weights for imbalance
# ---------------------------
def compute_class_weights(loader, num_classes, device='cuda'):
    """
    Computes class weights for imbalanced datasets.
    
    Args:
        loader (DataLoader): PyTorch DataLoader for training set
        num_classes (int): Number of classes
        device (str): 'cuda' or 'cpu'
    
    Returns:
        Tensor: Class weights to be used in CrossEntropyLoss
    """
    from collections import Counter
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float).to(device)

# ---------------------------
# FLOPs and parameters
# ---------------------------
def compute_flops_params(model, input_tensor):
    """
    Computes FLOPs and number of parameters for a single input.
    
    Returns:
        flops, params
    """
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params

# ---------------------------
# Latency measurement
# ---------------------------
def measure_latency(model, input_tensor, n=100, device='cuda'):
    """
    Measures average inference latency per image.
    
    Args:
        model (nn.Module): Model to evaluate
        input_tensor (Tensor): Single batch/input for measurement
        n (int): Number of iterations
        device (str): 'cuda' or 'cpu'
    
    Returns:
        float: Average latency per forward pass
    """
    model.eval()
    for _ in range(10): _ = model(input_tensor)  # warm-up
    if device=='cuda': torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n):
        _ = model(input_tensor)
    if device=='cuda': torch.cuda.synchronize()
    
    return (time.time() - start) / n


