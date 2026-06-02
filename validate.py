from __future__ import annotations

import argparse
import json
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


def eval_classes_for(cfg):
    return [1, 2, 3, 4, 5] if cfg["dataset"]["name"].lower() == "udd6" else None


def load_checkpoint_model(cfg, checkpoint, device, train_set):
    model = MEKDUAVSeg(cfg, class_balanced_weights(train_set.class_frequencies(), cfg["loss"]["class_weight_mu"])).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "student" in ckpt:
        model.deployed_student().load_state_dict(ckpt["student"], strict=True)
        model.eval()
        return model
    if "model" not in ckpt:
        raise KeyError("Checkpoint must contain either a full 'model' key or an exported 'student' key.")
    if ckpt.get("teacher_initialized", False):
        model.init_teacher_from_student()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_checkpoint(config, checkpoint):
    cfg = load_config(config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    _, loader, train_set, _ = build_loaders(cfg)
    model = load_checkpoint_model(cfg, checkpoint, device, train_set)

    eval_classes = eval_classes_for(cfg)
    metric = SegMetric(cfg["dataset"]["num_classes"], cfg["dataset"]["ignore_index"], eval_classes)
    for batch in tqdm(loader, desc="validate"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["mask"].to(device, non_blocking=True)
        metric.update(model.predict(images), labels)
    return metric.compute(), cfg


def main():
    parser = argparse.ArgumentParser(description="Validate the deployed MEKD-UAVSeg student.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default=None, help="Optional path for machine-readable metrics.")
    args = parser.parse_args()

    metrics, cfg = evaluate_checkpoint(args.config, args.checkpoint)
    print(metrics)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "dataset": cfg["dataset"]["name"],
            "split": cfg["dataset"]["val_split"],
            "metrics": metrics,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
