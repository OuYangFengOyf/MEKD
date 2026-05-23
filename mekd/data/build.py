from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader

from .datasets import UAVSegDataset


def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, UAVSegDataset, UAVSegDataset]:
    dcfg = cfg["dataset"]
    train_set = UAVSegDataset(
        root=dcfg["root"],
        split=dcfg["train_split"],
        dataset_name=dcfg["name"],
        image_size=dcfg["image_size"],
        num_classes=dcfg["num_classes"],
        ignore_index=dcfg["ignore_index"],
        train=True,
        label_mode=dcfg.get("label_mode", "auto"),
        palette=dcfg.get("palette"),
    )
    val_set = UAVSegDataset(
        root=dcfg["root"],
        split=dcfg["val_split"],
        dataset_name=dcfg["name"],
        image_size=dcfg["image_size"],
        num_classes=dcfg["num_classes"],
        ignore_index=dcfg["ignore_index"],
        train=False,
        label_mode=dcfg.get("label_mode", "auto"),
        palette=dcfg.get("palette"),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg["train"]["batch_size"] // 2),
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, train_set, val_set
