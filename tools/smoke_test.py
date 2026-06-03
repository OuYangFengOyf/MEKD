from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pass_msg(name: str) -> None:
    print(f"[PASS] {name}")


def fail_msg(name: str, exc: BaseException) -> None:
    print(f"[FAIL] {name}: {exc}")


def build_smoke_config():
    from config import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["seed"] = 123
    cfg["device"] = "cpu"
    cfg["dataset"]["num_classes"] = 6
    cfg["dataset"]["ignore_index"] = 255
    cfg["model"]["student"]["base_channels"] = 8
    cfg["model"]["student"]["fpn_channels"] = 32
    cfg["model"]["student"]["dropout"] = 0.0
    cfg["model"]["experts"]["channels"] = 64
    cfg["model"]["experts"]["transformer_layers"] = 1
    cfg["model"]["experts"]["transformer_heads"] = 4
    cfg["model"]["experts"]["mamba_blocks"] = 1
    cfg["train"]["prototype_start_epoch"] = 999
    cfg["train"]["uncertainty_start_epoch"] = 0
    cfg["train"]["distill_ramp_epochs"] = 1
    cfg["train"]["amp"] = False
    return cfg


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Sizes must use HxW format, for example 256x256.")
    h, w = value.lower().split("x", 1)
    return int(h), int(w)


def run_on_device(device_name: str, sizes: list[tuple[int, int]]) -> None:
    import torch

    from losses import class_balanced_weights
    from metrics import SegMetric
    from models import MEKDUAVSeg

    cfg = build_smoke_config()
    device = torch.device(device_name)
    num_classes = int(cfg["dataset"]["num_classes"])
    class_freq = torch.ones(num_classes) / num_classes
    class_weights = class_balanced_weights(class_freq, cfg["loss"]["class_weight_mu"])

    model = MEKDUAVSeg(cfg, class_weights).to(device)
    model.train()
    pass_msg(f"Model initialization ({device})")

    for batch_size, (height, width) in [(1, sizes[0]), (2, sizes[-1])]:
        images = torch.randn(batch_size, 3, height, width, device=device)
        labels = torch.randint(0, num_classes, (batch_size, height, width), device=device)
        model.zero_grad(set_to_none=True)
        out = model(images, labels, stage="warmup", epoch_in_stage=0, global_epoch=0)
        logits = out["logits"]
        if logits.shape != (batch_size, num_classes, height, width):
            raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
        pass_msg(f"Forward pass batch={batch_size} size={height}x{width} ({device})")
        pass_msg("Output shape")

        loss = out["loss"]
        if not torch.isfinite(loss):
            raise AssertionError("Loss is NaN or Inf.")
        pass_msg("Loss computation")
        loss.backward()
        pass_msg("Backward pass")

        if not torch.isfinite(logits.detach()).all():
            raise AssertionError("Logits contain NaN or Inf.")
        pass_msg("No NaN or Inf")

        metric = SegMetric(num_classes, ignore_index=255)
        metric.update(logits.detach(), labels.detach())
        metrics = metric.compute()
        for key in ["mPrecision", "mRecall", "mIoU", "mF1", "OA"]:
            if key not in metrics:
                raise AssertionError(f"Metric key missing: {key}")
        pass_msg("Metric computation")

    model.init_teacher_from_student()
    model.train()
    height, width = sizes[0]
    images = torch.randn(1, 3, height, width, device=device)
    labels = torch.randint(0, num_classes, (1, height, width), device=device)
    model.zero_grad(set_to_none=True)
    out = model(images, labels, stage="distill", epoch_in_stage=0, global_epoch=999)
    if out["logits"].shape != (1, num_classes, height, width):
        raise AssertionError(f"Unexpected distill logits shape: {tuple(out['logits'].shape)}")
    if not torch.isfinite(out["loss"]):
        raise AssertionError("Distillation loss is NaN or Inf.")
    out["loss"].backward()
    pass_msg(f"Distillation forward/backward ({device})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a random-tensor MEKD-UAVSeg smoke test.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[(256, 256), (512, 512), (255, 319)],
        help="Input sizes in HxW format.",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU smoke test even if CUDA is available.")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        fail_msg("PyTorch import", exc)
        return 1

    try:
        run_on_device("cpu", args.sizes)
        pass_msg("CPU smoke test")
        if torch.cuda.is_available() and not args.cpu_only:
            run_on_device("cuda", args.sizes[:1])
            pass_msg("GPU smoke test")
        elif not args.cpu_only:
            print("[SKIP] GPU smoke test: CUDA is not available")
    except Exception as exc:
        fail_msg("Smoke test", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
