# utils/metrics.py
import time
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile

def compute_classification_metrics(y_true, y_pred, class_names):
    """
    Prints classification report and plots confusion matrix.
    Returns accuracy and macro-F1 score.
    """
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    acc = sum([p==t for p,t in zip(y_pred, y_true)])/len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return acc, macro_f1

def measure_latency(model, input_tensor, n=100, device='cuda'):
    """
    Measures average inference latency per image on CPU or GPU.
    """
    model.eval()
    for _ in range(10): _ = model(input_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n):
        _ = model(input_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start)/n

def compute_flops_params(model, input_tensor):
    """
    Computes FLOPs and number of parameters for a single input.
    """
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params
