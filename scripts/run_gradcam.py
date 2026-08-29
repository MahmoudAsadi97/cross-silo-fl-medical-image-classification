#!/usr/bin/env python3
"""Bonus: Grad-CAM explainability on Fed-ISIC2019.

Overlays a Grad-CAM saliency map (Selvaraju et al., 2017) on dermoscopy test images using
the federated global model, showing WHERE the model looks when it classifies a lesion — an
interpretability / clinical-trust check. Uses the last convolutional block (``layer4``).
Runs at a larger input size than training so the heat-map has usable resolution (the
classifier head is size-agnostic thanks to adaptive average pooling).

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/run_gradcam.py \
        --model experiments/live/pretrained.pt --device cuda --n 8 --image-size 224
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402

# ISIC-2019 diagnostic classes, in the dataset's class_0..class_7 order.
CLASS_NAMES = ["Melanoma", "Melanocytic nevus", "Basal cell carcinoma", "Actinic keratosis",
               "Benign keratosis", "Dermatofibroma", "Vascular lesion", "Squamous cell carcinoma"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(REPO / "experiments" / "live" / "pretrained.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=8, help="number of images (aim: one per class)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import matplotlib
    import numpy as np
    import torch
    import torch.nn.functional as F

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fl_med.data.dataset import ISICFederatedFolderDataset
    from fl_med.data.paths import resolve_data_root
    from fl_med.data.transforms import get_eval_transforms
    from fl_med.models import build_model

    overrides = [f"data.image_size={args.image_size}", "data.num_workers=0"]
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if data_root:
        overrides.append(f"data.root={data_root}")
    config = resolve_config(REPO / "configs" / "fedavg.yaml", tier="dev", overrides=overrides)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    model = build_model(config).to(device).eval()
    if Path(args.model).exists():
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"loaded model: {args.model}")
    else:
        print(f"WARNING: {args.model} not found — using an untrained model (run pretrain first)")

    root = Path(resolve_data_root(config)) / "test"
    ds = ISICFederatedFolderDataset(root, transform=get_eval_transforms(args.image_size))
    rng = np.random.default_rng(args.seed)
    by_class: dict = {}
    for idx, (_, cls, _) in enumerate(ds.samples):
        by_class.setdefault(cls, []).append(idx)
    picks = [int(rng.choice(by_class[c])) for c in sorted(by_class)][:args.n]
    while len(picks) < args.n:
        picks.append(int(rng.integers(len(ds))))

    # Grad-CAM hooks on the last conv block.
    acts, grads = {}, {}
    layer = model.layer4
    layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))

    ncol = 4
    nrow = (args.n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 3.3))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, picks):
        item = ds[idx]
        x = item["image"].unsqueeze(0).to(device)
        true = int(item["label"])
        acts.clear()
        grads.clear()
        logits = model(x)
        pred = int(logits.argmax(1))
        model.zero_grad()
        logits[0, pred].backward()
        A, G = acts["v"][0], grads["v"][0]           # (C,h,w)
        weights = G.mean(dim=(1, 2))                 # (C,)
        cam = torch.relu((weights[:, None, None] * A).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(args.image_size,) * 2,
                            mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        img = x[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.axis("off")
        mark = "✓" if pred == true else "✗"
        ax.set_title(f"pred: {CLASS_NAMES[pred][:18]}\ntrue: {CLASS_NAMES[true][:18]} [{mark}]",
                     fontsize=7)
    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle("Grad-CAM — where the federated model looks when classifying skin lesions",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPO / "reports" / "figures" / "gradcam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
