import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

class ResNetClassifier:
    def __init__(self, num_classes, weights=None, freeze_backbone=True):
        """
        Initialize the ResNet classifier for PlantVillage dataset.
        
        Args:
            num_classes (int): Number of output classes
            weights (torch.Tensor): Optional class weights for imbalance handling
            freeze_backbone (bool): Whether to freeze ResNet backbone weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes, freeze_backbone)
        self.criterion = nn.CrossEntropyLoss(weight=weights) if weights is not None else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters())
        
    def _build_model(self, num_classes, freeze_backbone):
        """Build ResNet model"""
        model = models.resnet50(pretrained=True)
        
        if freeze_backbone:
            # Freeze all layers except the final classifier
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze the final classifier
            for param in model.fc.parameters():
                param.requires_grad = True
                
        # Build appropriate classifier based on number of classes
        if num_classes <= 2:  # binary
            model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        else:  #multi
            model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
            
        return model.to(self.device)
    
    def get_model(self):
        """Return the configured model"""
        return self.model
    
    def get_criterion(self):
        """Return the loss function"""
        return self.criterion
    
    def get_optimizer(self):
        """Return the optimizer"""
        return self.optimizer
    
    def get_device(self):
        """Return the device being used"""
        return self.device