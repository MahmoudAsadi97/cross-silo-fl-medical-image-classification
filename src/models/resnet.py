from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
