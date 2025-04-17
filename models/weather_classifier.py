# models/weather_classifier.py
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

NUM_WEATHER_CLASSES = 6  # clear, partly cloudy, overcast, rainy, snowy, foggy

class WeatherClassifier(nn.Module):
    """
    ConvNeXt‑Tiny backbone → 6‑class image‑level weather classifier.
    """
    def __init__(self, num_classes: int = NUM_WEATHER_CLASSES, freeze_backbone: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = convnext_tiny(weights=weights)
        
        # Optionally freeze backbone for faster convergence
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_feats = self.backbone.classifier[2].in_features   # 768
        # Replace last linear layer
        self.backbone.classifier[2] = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)
