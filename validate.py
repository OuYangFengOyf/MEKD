from __future__ import annotations

import argparse
import json
from pathlib import Path


def eval_classes_for(cfg):
    return [1, 2, 3, 4, 5] if cfg["dataset"]["name"].lower() == "udd6" else None


def _load_runtime():
    global torch, tqdm, load_config, build_eval_loader, class_balanced_weights, SegMetric, MEKDUAVSeg

    import torch
    from tqdm import tqdm

    from config import load_config
    from data import build_eval_loader
    from losses import class_balanced_weights
    from metrics import SegMetric
    from models import MEKDUAVSeg


def resolve_device(cfg):
    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_checkpoint_model(cfg, checkpoint, device):
    num_classes = int(cfg["dataset"]["num_classes"])
    uniform_freq = torch.ones(num_classes, dtype=torch.float32) / num_classes
    weights = class_balanced_weights(uniform_freq, cfg["loss"]["class_weight_mu"])
    model = MEKDUAVSeg(cfg, weights).to(device)

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
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


def evaluate_checkpoint(config, checkpoint, split=None):
    _load_runtime()
    cfg = load_config(config)
    device = resolve_device(cfg)
    loader, _ = build_eval_loader(cfg, split=split)
    model = load_checkpoint_model(cfg, checkpoint, device)

    eval_classes = eval_classes_for(cfg)
    metric = SegMetric(cfg["dataset"]["num_classes"], cfg["dataset"]["ignore_index"], eval_classes)
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True)
            metric.update(model.predict(images), labels)
    return metric.compute(), cfg


def main():
    parser = argparse.ArgumentParser(description="Validate or test the deployed MEKD-UAVSeg student.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=None, help="Optional split override, e.g. val or test.")
    parser.add_argument("--output-json", default=None, help="Optional path for machine-readable metrics.")
    args = parser.parse_args()

    metrics, cfg = evaluate_checkpoint(args.config, args.checkpoint, split=args.split)
    print(metrics)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "dataset": cfg["dataset"]["name"],
            "split": args.split or cfg["dataset"]["val_split"],
            "metrics": metrics,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
