"""Model builders + a normalization converter for DP.

* ``SmallCNN``          -- tiny, torchvision-free net for the ``smoke`` tier and
                          CI (no pretrained-weight download needed).
* ``build_resnet18``    -- FLamby-style backbone; ``norm='group'`` swaps BatchNorm
                          for GroupNorm so the *non-private* baseline can be trained
                          on the SAME architecture used under DP (Opacus cannot do
                          per-sample grads through BatchNorm). This isolates the DP
                          effect from the normalization effect (brief §4.1).
* ``build_mobilenet_v2`` / ``build_efficientnet_b0`` -- edge / FLamby-reference nets.
* ``build_model(config)`` -- factory driven by the ``model`` config block.

All torch/torchvision imports are lazy so the package imports without them.
"""
from __future__ import annotations

from typing import Any, Dict


def convert_bn_to_gn(model, num_groups: int = 8):
    """Recursively replace every ``BatchNorm2d`` with a ``GroupNorm``.

    Group count is clamped to divide the channel count (falls back to 1 group =
    LayerNorm-like) so it is valid for any layer width.
    """
    import torch.nn as nn

    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = num_groups
            while groups > 1 and num_channels % groups != 0:
                groups -= 1
            setattr(model, name, nn.GroupNorm(groups, num_channels, affine=True))
        else:
            convert_bn_to_gn(child, num_groups)
    return model


def build_small_cnn(num_classes: int = 8, norm: str = "group", **_: Any):
    """A compact CNN used for smoke tests. GroupNorm by default (DP-friendly)."""
    import torch.nn as nn

    def norm_layer(c: int):
        if norm == "group":
            g = 8
            while g > 1 and c % g != 0:
                g -= 1
            return nn.GroupNorm(g, c)
        if norm == "batch":
            return nn.BatchNorm2d(c)
        return nn.Identity()

    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), norm_layer(16), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), norm_layer(32), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return SmallCNN()


def build_resnet18(num_classes: int = 8, pretrained: bool = True, norm: str = "batch", **_: Any):
    import torch.nn as nn
    from torchvision import models

    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:  # very old torchvision
        model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if norm == "group":
        model = convert_bn_to_gn(model)
    return model


def build_mobilenet_v2(num_classes: int = 8, pretrained: bool = True, norm: str = "batch", **_: Any):
    import torch.nn as nn
    from torchvision import models

    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    if norm == "group":
        model = convert_bn_to_gn(model)
    return model


def build_efficientnet_b0(
    num_classes: int = 8, pretrained: bool = True, norm: str = "batch", **_: Any
):
    """FLamby's reference backbone (EfficientNet-B0, ImageNet-pretrained)."""
    import torch.nn as nn
    from torchvision import models

    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    if norm == "group":
        model = convert_bn_to_gn(model)
    return model


_BUILDERS = {
    "small_cnn": build_small_cnn,
    "resnet18": build_resnet18,
    "mobilenet_v2": build_mobilenet_v2,
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(config: Dict[str, Any]):
    """Factory from a config ``model`` block: ``{name, num_classes, pretrained, norm}``."""
    model_cfg = dict(config.get("model", {}) or {})
    name = model_cfg.pop("name", "small_cnn")
    if name not in _BUILDERS:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(_BUILDERS)}")
    return _BUILDERS[name](**model_cfg)


def model_builder_from_config(config: Dict[str, Any]):
    """Return a zero-arg callable that constructs a fresh model (for FL clients)."""
    return lambda: build_model(config)
