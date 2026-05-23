from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mekd.config import load_config
from mekd.data import build_loaders
from mekd.losses import class_balanced_weights
from mekd.metrics import SegMetric
from mekd.models import MEKDUAVSeg


def main():
    parser = argparse.ArgumentParser(description="Validate the deployed MEKD-UAVSeg student.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    _, loader, train_set, _ = build_loaders(cfg)
    model = MEKDUAVSeg(cfg, class_balanced_weights(train_set.class_frequencies(), cfg["loss"]["class_weight_mu"])).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if ckpt.get("teacher_initialized", False):
        model.init_teacher_from_student()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    eval_classes = [0, 1, 2, 3, 4] if cfg["dataset"]["name"].lower() == "udd6" else None
    metric = SegMetric(cfg["dataset"]["num_classes"], cfg["dataset"]["ignore_index"], eval_classes)
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True)
            metric.update(model.predict(images), labels)
    print(metric.compute())


if __name__ == "__main__":
    main()
