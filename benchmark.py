from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the published MEKD-UAVSeg validation protocol.")
    parser.add_argument("--config", required=True, help="Path to the exact YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Full MEKD checkpoint or exported student checkpoint.")
    parser.add_argument("--output", required=True, help="JSON file that records metrics and protocol metadata.")
    parser.add_argument("--run-name", default=None, help="Optional human-readable run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    from validate import evaluate_checkpoint

    metrics, cfg = evaluate_checkpoint(args.config, args.checkpoint)
    payload = {
        "run_name": args.run_name or Path(args.checkpoint).stem,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_entrypoint": "python benchmark.py",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "dataset": {
            "name": cfg["dataset"]["name"],
            "root": cfg["dataset"]["root"],
            "train_split": cfg["dataset"]["train_split"],
            "val_split": cfg["dataset"]["val_split"],
            "image_size": cfg["dataset"]["image_size"],
            "num_classes": cfg["dataset"]["num_classes"],
            "ignore_index": cfg["dataset"]["ignore_index"],
            "note": "Raw UAV imagery and labels are not redistributed by this repository; use the official dataset access channels.",
        },
        "training_protocol": {
            "seed": cfg["seed"],
            "epochs": cfg["train"]["epochs"],
            "warmup_epochs": cfg["train"]["warmup_epochs"],
            "prototype_start_epoch": cfg["train"]["prototype_start_epoch"],
            "uncertainty_start_epoch": cfg["train"]["uncertainty_start_epoch"],
            "batch_size": cfg["train"]["batch_size"],
            "learning_rate": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "amp": cfg["train"]["amp"],
        },
        "evaluation_protocol": {
            "metrics": ["mIoU", "mF1", "OA", "per-class IoU", "per-class F1"],
            "udd6_classes": "For UDD6, mIoU/mF1 average foreground classes 1-5 and exclude class 0 (Other).",
            "checkpoint_type": "full MEKD checkpoint or exported deployable student checkpoint.",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "metrics": metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
