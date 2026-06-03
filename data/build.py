from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader

from .datasets import UAVSegDataset
from utils import make_torch_generator, make_worker_init_fn


def build_dataset(cfg: Dict, split: str, train: bool) -> UAVSegDataset:
    dcfg = cfg["dataset"]
    return UAVSegDataset(
        root=dcfg["root"],
        split=split,
        dataset_name=dcfg["name"],
        image_size=dcfg["image_size"],
        num_classes=dcfg["num_classes"],
        ignore_index=dcfg["ignore_index"],
        train=train,
        label_mode=dcfg.get("label_mode", "auto"),
        palette=dcfg.get("palette"),
    )


def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, UAVSegDataset, UAVSegDataset]:
    dcfg = cfg["dataset"]
    tcfg = cfg["train"]
    seed = int(cfg.get("seed", 0))
    train_set = build_dataset(cfg, dcfg["train_split"], train=True)
    val_set = build_dataset(cfg, dcfg["val_split"], train=False)
    batch_size = int(tcfg["batch_size"])
    drop_last = bool(tcfg.get("drop_last", True)) and len(train_set) >= batch_size
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=bool(tcfg.get("pin_memory", True)),
        drop_last=drop_last,
        worker_init_fn=make_worker_init_fn(seed),
        generator=make_torch_generator(seed),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, int(tcfg.get("eval_batch_size", batch_size // 2))),
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=bool(tcfg.get("pin_memory", True)),
        worker_init_fn=make_worker_init_fn(seed),
        generator=make_torch_generator(seed),
    )
    return train_loader, val_loader, train_set, val_set


def build_eval_loader(cfg: Dict, split: str | None = None) -> Tuple[DataLoader, UAVSegDataset]:
    dcfg = cfg["dataset"]
    tcfg = cfg["train"]
    seed = int(cfg.get("seed", 0))
    dataset = build_dataset(cfg, split or dcfg["val_split"], train=False)
    batch_size = int(tcfg["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(tcfg.get("eval_batch_size", batch_size // 2))),
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=bool(tcfg.get("pin_memory", True)),
        worker_init_fn=make_worker_init_fn(seed),
        generator=make_torch_generator(seed),
    )
    return loader, dataset
