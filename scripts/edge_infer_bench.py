#!/usr/bin/env python3
"""Edge INFERENCE benchmark: how fast can the trained model classify a lesion on
a small device — and does quantization help?

Important distinction (defense-ready): quantization/pruning speed up *inference*,
not federated *training* (training needs float gradients; the training speedup is
freeze-backbone — see bench_pi.py). This script benchmarks the deployment side:

  variants (each fail-soft — skipped with a reason if unsupported on this build):
    fp32          — plain eager model (baseline)
    torchscript   — JIT-traced fp32 (graph optimizations)
    dynamic-int8  — dynamic quantization (Linear layers only; convs stay fp32,
                    so expect a small effect on a CNN — reported honestly)
    static-int8   — FX-mode post-training static quantization of the whole net
                    (convs included; calibrated on local data; biggest win if
                    the build supports GroupNorm quantization)

  metrics: median ms/image (batch 1), model size (MB), balanced accuracy on this
  device's local test shard (so we can SEE any accuracy cost of quantization).

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/edge_infer_bench.py \
        --client-id 5 --model experiments/live/pretrained.pt

Writes experiments/live/edge_infer_bench.json.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402


def model_size_mb(obj) -> float:
    import torch

    buf = io.BytesIO()
    try:
        torch.jit.save(obj, buf)          # scripted/traced modules
    except Exception:
        torch.save(obj.state_dict(), buf)  # eager modules
    return round(buf.getbuffer().nbytes / 1e6, 2)


def bench_latency(fn, x, warmup=3, iters=20):
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(ts), 1)


def accuracy(fn, loader, num_classes=8):
    import torch

    from fl_med.metrics import balanced_accuracy

    ys, ps = [], []
    with torch.inference_mode():
        for batch in loader:
            out = fn(batch["image"])
            ps.extend(torch.argmax(out, 1).tolist())
            ys.extend(batch["label"].tolist())
    return round(balanced_accuracy(ys, ps, num_classes), 4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, default=5)
    p.add_argument("--model", default=str(REPO / "experiments" / "live" / "pretrained.pt"))
    p.add_argument("--image-size", type=int, default=64,
                   help="use the model's training resolution")
    p.add_argument("--data-root", default=None)
    args = p.parse_args(argv)

    import torch

    from fl_med.data.loaders import build_client_dataloaders
    from fl_med.federated_live import build_model

    torch.set_num_threads(os.cpu_count() or 4)
    is_arm = platform.machine().lower().startswith(("aarch64", "arm"))
    engine = "qnnpack" if is_arm else "fbgemm"
    try:
        torch.backends.quantized.engine = engine
    except Exception:
        pass

    overrides = ["data.num_workers=0", f"data.image_size={args.image_size}"]
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if data_root:
        overrides.append(f"data.root={data_root}")
    config = resolve_config(REPO / "configs" / "live_fedavg.yaml", tier="dev", overrides=overrides)

    model = build_model(config).eval()
    mp = Path(args.model)
    if mp.exists():
        model.load_state_dict(torch.load(mp, map_location="cpu"))
        print(f"loaded {mp}")
    else:
        print(f"WARNING: {mp} missing — benchmarking an untrained model")

    train_loader, test_loader = build_client_dataloaders(config, args.client_id)
    x = torch.randn(1, 3, args.image_size, args.image_size)

    results = {"host": platform.node(), "machine": platform.machine(),
               "threads": torch.get_num_threads(), "quant_engine": engine,
               "image_size": args.image_size, "variants": {}}

    def record(name, fn, size_obj):
        try:
            lat = bench_latency(fn, x)
            acc = accuracy(fn, test_loader)
            results["variants"][name] = {"ms_per_image": lat, "size_mb": model_size_mb(size_obj),
                                         "balanced_accuracy": acc}
            print(f"  {name:12} {lat:7.1f} ms/img   {results['variants'][name]['size_mb']:6.2f} MB"
                  f"   bal_acc={acc}")
        except Exception as e:  # noqa: BLE001
            results["variants"][name] = {"skipped": f"{type(e).__name__}: {e}"}
            print(f"  {name:12} SKIPPED ({type(e).__name__}: {e})")

    print(f"\n=== edge inference benchmark ({platform.machine()}, "
          f"{torch.get_num_threads()} threads, engine={engine}) ===")

    # 1. fp32 eager
    record("fp32", model, model)

    # 2. TorchScript
    try:
        ts_model = torch.jit.trace(model, x)
        ts_model = torch.jit.freeze(ts_model.eval())
        record("torchscript", ts_model, ts_model)
    except Exception as e:  # noqa: BLE001
        results["variants"]["torchscript"] = {"skipped": str(e)}
        print(f"  torchscript  SKIPPED ({e})")

    # 3. dynamic int8 (Linear only — honest about the small effect on a CNN)
    try:
        dyn = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        record("dynamic-int8", dyn, dyn)
    except Exception as e:  # noqa: BLE001
        results["variants"]["dynamic-int8"] = {"skipped": str(e)}
        print(f"  dynamic-int8 SKIPPED ({e})")

    # 4. static int8 via FX graph mode (convs included; needs calibration)
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        qmap = get_default_qconfig_mapping(engine)
        prepared = prepare_fx(build_model(config).eval(), qmap, example_inputs=(x,))
        prepared.load_state_dict(model.state_dict(), strict=False)
        with torch.inference_mode():          # calibrate on a few local batches
            for i, batch in enumerate(train_loader):
                prepared(batch["image"])
                if i >= 7:
                    break
        static = convert_fx(prepared)
        record("static-int8", static, static)
    except Exception as e:  # noqa: BLE001
        results["variants"]["static-int8"] = {"skipped": f"{type(e).__name__}: {e}"}
        print(f"  static-int8  SKIPPED ({type(e).__name__}: {e})")

    out = REPO / "experiments" / "live" / "edge_infer_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
