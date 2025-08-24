import torch
import torch.nn as nn
from torchvision.models import vgg16
from timm import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LeafNet(nn.Module):
    def __init__(self, n_class=14):
        super(LeafNet, self).__init__() 

        vgg = vgg16(pretrained=True)

        self.efficient = create_model(
            'tf_efficientnetv2_l',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

        # Freeze early VGG layers
        for param in vgg.features[:12].parameters():
            param.requires_grad = False

        # VGG outputs 256 channels at layer conv3_1 (index 12)
        self.vgg_features = nn.Sequential(*list(vgg.features[:12]))

        # Adapt output to expected input for EfficientNet (must match 3 channels)
        self.adapt = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )

        # EfficientNet-v2-L outputs 1280 features
        self.classifier = nn.Sequential(
            nn.Linear(1280, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_class)
        )

        self.vgg = vgg  # Store for access in `unfreeze`

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.adapt(x)
        x = self.efficient(x)
        x = self.classifier(x)
        return x

    def unfreeze(self):
        for param in self.vgg.features[:12].parameters():
            param.requires_grad = True
