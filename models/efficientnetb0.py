import torch
import torch.nn as nn
from torchvision import models

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=38, freeze_backbone=True):
        super().__init__()
        model = models.efficientnet_b0(pretrained=True)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.model = model
        
    def forward(self, x):
        return self.model(x)
