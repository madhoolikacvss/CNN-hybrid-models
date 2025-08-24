import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

class MobileNetV2Classifier:
    def __init__(self, num_classes, weights=None, freeze_backbone=True):
        """
        Initialize the MobileNetV2 classifier for PlantVillage dataset.
        
        Args:
            num_classes (int): Number of output classes
            weights (torch.Tensor): Optional class weights for imbalance handling
            freeze_backbone (bool): Whether to freeze MobileNetV2 backbone weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes, freeze_backbone)
        self.criterion = nn.CrossEntropyLoss(weight=weights) if weights is not None else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters())
        
    def _build_model(self, num_classes, freeze_backbone):
        """Build MobileNetV2 model"""
        model = models.mobilenet_v2(pretrained=True)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
                
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
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