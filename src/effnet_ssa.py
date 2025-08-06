import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

class selfSpatialAttention(nn.Module):
    def __init__(self, in_channnels):
        super().__init__()
        self.conv = nn.Conv2d(in_channnels, 1, kernel_size=1)

    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn


class EffNetB3_SSA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT
        eff = models.efficientnet_b3(weights=weights)
        self.features = eff.features
        channels = eff.classifier[1].in_features
        self.attn = selfSpatialAttention(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.pool(x)
        return self.mlp(x)