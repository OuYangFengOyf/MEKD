from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cuda",
    "dataset": {
        "name": "uavid",
        "root": "data/UAVid",
        "train_split": "train",
        "val_split": "val",
        "image_size": [512, 1024],
        "num_classes": 8,
        "ignore_index": 255,
        "label_mode": "auto",
        "palette": None,
        "small_object_quantile": 0.3,
    },
    "model": {
        "student": {
            "base_channels": 32,
            "fpn_channels": 128,
            "dropout": 0.1,
        },
        "experts": {
            "channels": 256,
            "transformer_layers": 2,
            "transformer_heads": 8,
            "mamba_blocks": 2,
        },
    },
    "train": {
        "epochs": 160,
        "warmup_epochs": 40,
        "prototype_start_epoch": 10,
        "uncertainty_start_epoch": 10,
        "batch_size": 16,
        "num_workers": 4,
        "lr": 0.0004,
        "weight_decay": 0.01,
        "warmup_iters": 500,
        "distill_ramp_epochs": 5,
        "save_dir": "runs/mekd_uavid",
        "amp": True,
    },
    "loss": {
        "lambda_dice": 0.5,
        "lambda_t": 1.0,
        "lambda_m": 1.0,
        "lambda_proto": 0.05,
        "lambda_den": 0.1,
        "lambda_ms": 0.5,
        "lambda_logit": 1.0,
        "lambda_bd": 0.2,
        "lambda_route": 0.05,
        "temperature_proto": 0.2,
        "temperature_logit": 4.0,
        "temperature_gate": 0.2,
        "temperature_route": 1.0,
        "hard_region_beta": 0.5,
        "prototype_momentum": 0.99,
        "prototype_min_pixels": 100,
        "ema_threshold_momentum": 0.95,
        "class_weight_mu": 1.02,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULT_CONFIG, user_cfg)
    cfg["config_path"] = str(path)
    return cfg
