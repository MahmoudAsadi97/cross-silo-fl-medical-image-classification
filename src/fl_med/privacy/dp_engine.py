"""Opacus DP-SGD engine wrapper (torch; imported lazily).

Local (per-client) DP-SGD: clip each per-sample gradient to ``max_grad_norm`` and
add Gaussian noise scaled by ``noise_multiplier``, giving SAMPLE-level (record-level)
DP for that client's own training. NOT client-level DP (see design notes).

``fix_model`` makes a torchvision ResNet Opacus-compatible:
  1. GroupNorm instead of BatchNorm (Opacus needs per-sample grads; ours is already GN),
  2. no in-place activations (nn.ReLU(inplace=True) modifies Opacus hook views),
  3. non-in-place residual add (BasicBlock/Bottleneck use ``out += identity``, also an
     in-place op on a hook view). All three raise
     "a view is being modified inplace ... forbidden" otherwise (brief §4.1-4.2).
"""
from __future__ import annotations

import types
from typing import Any, Tuple


def _disable_inplace(model):
    import torch.nn as nn

    inplace_types = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.SiLU, nn.Hardswish)
    for module in model.modules():
        if isinstance(module, inplace_types) and getattr(module, "inplace", False):
            module.inplace = False
    return model


def _basicblock_forward(self, x):
    identity = x
    out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
    out = self.conv2(out); out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity            # non-in-place (was: out += identity)
    return self.relu(out)


def _bottleneck_forward(self, x):
    identity = x
    out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
    out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
    out = self.conv3(out); out = self.bn3(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    return self.relu(out)


def _patch_residuals(model):
    """Replace in-place residual adds in torchvision ResNet blocks."""
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except Exception:
        return model
    for m in model.modules():
        if isinstance(m, BasicBlock):
            m.forward = types.MethodType(_basicblock_forward, m)
        elif isinstance(m, Bottleneck):
            m.forward = types.MethodType(_bottleneck_forward, m)
    return model


def fix_model(model):
    """Make a model DP-compatible: GroupNorm + no in-place activations/residuals."""
    from opacus.validators import ModuleValidator

    model = _disable_inplace(model)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        _disable_inplace(model)
    return _patch_residuals(model)


def make_private(
    *, model, optimizer, data_loader, noise_multiplier: float, max_grad_norm: float,
) -> Tuple[Any, Any, Any, Any]:
    """Attach an Opacus ``PrivacyEngine``. Returns (model, optimizer, loader, engine)."""
    from opacus import PrivacyEngine

    engine = PrivacyEngine()
    model, optimizer, data_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=data_loader,
        noise_multiplier=float(noise_multiplier), max_grad_norm=float(max_grad_norm),
        poisson_sampling=True,
    )
    return model, optimizer, data_loader, engine


def opacus_epsilon(engine, delta: float) -> float:
    try:
        return float(engine.get_epsilon(delta))
    except Exception:
        return float(engine.accountant.get_epsilon(delta=delta))
