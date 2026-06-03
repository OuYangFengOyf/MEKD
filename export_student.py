from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export only the deployable STDC2-FPN student.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    from config import load_config
    from losses import class_balanced_weights
    from models import MEKDUAVSeg

    cfg = load_config(args.config)
    weights = torch.ones(cfg["dataset"]["num_classes"])
    model = MEKDUAVSeg(cfg, class_balanced_weights(weights / weights.sum(), cfg["loss"]["class_weight_mu"]))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    if ckpt.get("teacher_initialized", False):
        model.init_teacher_from_student()
    model.load_state_dict(ckpt["model"], strict=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": model.deployed_student().state_dict(), "config": cfg}, output)


if __name__ == "__main__":
    main()
