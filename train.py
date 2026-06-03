from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train MEKD-UAVSeg.")
    parser.add_argument("--config", required=True, help="Path to a MEKD-UAVSeg YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume.")
    return parser.parse_args()


def _load_runtime():
    global torch, GradScaler, autocast, tqdm
    global load_config, build_loaders, class_balanced_weights, SegMetric, MEKDUAVSeg, set_seed

    import torch
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm

    from config import load_config
    from data import build_loaders
    from losses import class_balanced_weights
    from metrics import SegMetric
    from models import MEKDUAVSeg
    from utils import set_seed


def resolve_device(cfg):
    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main():
    args = parse_args()
    _load_runtime()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]), deterministic=bool(cfg["train"].get("deterministic", False)))
    device = resolve_device(cfg)
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_set, _ = build_loaders(cfg)
    freq = train_set.class_frequencies()
    cfg["dataset"]["small_object_threshold"] = train_set.connected_component_area_quantile(
        float(cfg["dataset"].get("small_object_quantile", 0.3))
    )
    (save_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    class_weights = class_balanced_weights(freq, float(cfg["loss"]["class_weight_mu"]))
    model = MEKDUAVSeg(cfg, class_weights).to(device)

    start_epoch = 0
    best_miou = -1.0
    ckpt = None
    if args.resume:
        resume = Path(args.resume)
        if not resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        ckpt = torch.load(resume, map_location="cpu")
        if ckpt.get("teacher_initialized", False):
            model.init_teacher_from_student()
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt["epoch"]) + 1
        best_miou = float(ckpt.get("best_miou", -1.0))

    optimizer = make_optimizer(model, cfg)
    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    scheduler = make_scheduler(optimizer, cfg, len(train_loader), start_epoch)
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    warmup_epochs = int(cfg["train"]["warmup_epochs"])
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        if epoch == int(cfg["train"]["prototype_start_epoch"]):
            initialize_prototypes(model, train_loader, device, epoch)

        if epoch == warmup_epochs:
            model.init_teacher_from_student()
            optimizer = make_optimizer(model, cfg)
            scheduler = make_scheduler(optimizer, cfg, len(train_loader), epoch)

        stage = "warmup" if epoch < warmup_epochs else "distill"
        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch, stage, warmup_epochs)
        metrics = validate(model, val_loader, cfg, device)
        is_best = metrics["mIoU"] > best_miou
        best_miou = max(best_miou, metrics["mIoU"])
        save_checkpoint(save_dir / "last.pth", model, optimizer, epoch, best_miou)
        if is_best:
            save_checkpoint(save_dir / "best.pth", model, optimizer, epoch, best_miou)
        print(f"epoch={epoch:03d} stage={stage} mIoU={metrics['mIoU']:.2f} mF1={metrics['mF1']:.2f} OA={metrics['OA']:.2f}")


def make_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))


def make_scheduler(optimizer, cfg, iters_per_epoch: int, start_epoch: int):
    total_iters = max(1, int(cfg["train"]["epochs"]) * iters_per_epoch)
    warmup_iters = int(cfg["train"]["warmup_iters"])
    start_iter = start_epoch * iters_per_epoch

    def lr_lambda(step):
        cur = start_iter + step
        if cur < warmup_iters:
            return max(1e-6, float(cur + 1) / max(1, warmup_iters))
        progress = (cur - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, epoch, stage, warmup_epochs):
    model.train()
    if stage == "distill":
        model.transformer_semantic_expert.eval()
        model.mamba_spatial_expert.eval()
        model.structure_density_predictor.eval()
        model.teacher_side_encoder.eval()

    pbar = tqdm(loader, desc=f"{stage} {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
            out = model(images, labels, stage=stage, epoch_in_stage=max(0, epoch - warmup_epochs), global_epoch=epoch)
            loss = out["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if stage == "warmup" and epoch >= int(cfg["train"]["prototype_start_epoch"]):
            model.update_prototypes(out["t_feat"], labels)
        pbar.set_postfix(loss=float(loss.detach().cpu()))


def initialize_prototypes(model, loader, device, epoch):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="init prototypes", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True)
            out = model(images, labels, stage="warmup", epoch_in_stage=epoch, global_epoch=epoch)
            model.update_prototypes(out["t_feat"], labels)


def validate(model, loader, cfg, device):
    model.eval()
    eval_classes = [1, 2, 3, 4, 5] if cfg["dataset"]["name"].lower() == "udd6" else list(range(cfg["dataset"]["num_classes"]))
    metric = SegMetric(cfg["dataset"]["num_classes"], cfg["dataset"]["ignore_index"], eval_classes)
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True)
            metric.update(model.predict(images), labels)
    return metric.compute()


def save_checkpoint(path, model, optimizer, epoch, best_miou):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
            "teacher_initialized": model.teacher_side_encoder is not None,
        },
        path,
    )


if __name__ == "__main__":
    main()
