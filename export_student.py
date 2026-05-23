from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mekd.config import load_config
from mekd.losses import class_balanced_weights
from mekd.models import MEKDUAVSeg


def main():
    parser = argparse.ArgumentParser(description="Export only the deployable STDC2-FPN student.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = torch.ones(cfg["dataset"]["num_classes"])
    model = MEKDUAVSeg(cfg, class_balanced_weights(weights / weights.sum(), cfg["loss"]["class_weight_mu"]))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if ckpt.get("teacher_initialized", False):
        model.init_teacher_from_student()
    model.load_state_dict(ckpt["model"], strict=True)
    torch.save({"student": model.deployed_student().state_dict(), "config": cfg}, args.output)


if __name__ == "__main__":
    main()
