'''
    this hybrid tries to use MobilenetV2 as a fast ealry feature extractor with EfficientNetV2-B0 a a deeper sementic feature extractor
    I will combine them with a adapter block 
    replast head/classifier to suit 14 classes '''
    
import torch
import torch.nn as nn
import torchvision.models as models
import timm


class LeafNetv2(nn.Module):
    def __init__(self, n_class=14):
        super(LeafNetv2, self).__init__()

        mobilenet = models.mobilenet_v2(pretrained=True)
        self.mnet_features = nn.Sequential(*mobilenet.features[:7])  # up to bottleneck 6

        # Freeze MobileNet early layers to not change weights
        for param in self.mnet_features.parameters():
            param.requires_grad = True

        self.adapter = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),    # 16 tp 3
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False) #resizes inp dims cause mob output at 7 is 28x28x64 btu eff expects 224x224x3
            #bilinear is an interpolation method good for images when upsamplping
        )

        # effnetv2 backbone
        self.effnet = timm.create_model(
            'tf_efficientnetv2_b0',
            pretrained=True,
            num_classes=0,             # no classifier head
            global_pool='avg'          # output shape:1280
        )

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_class)
        )

    def forward(self, x):
        x = self.mnet_features(x)      #64, 28, 28
        x = self.adapter(x)            #, 224, 224
        x = self.effnet(x)            # 1280
        x = self.classifier(x)        #  n_class
        return x

