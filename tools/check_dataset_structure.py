from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_IMAGE_DIRS = ("images", "image", "imgs", "img", "Images")
DEFAULT_LABEL_DIRS = ("labels", "label", "masks", "mask", "gt", "ann", "Labels")


def parse_args():
    parser = argparse.ArgumentParser(description="Check UAV segmentation dataset structure without modifying data.")
    parser.add_argument("--root", required=True, help="Dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to check, e.g. train val test.")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--palette", default=None, help="Optional YAML config containing dataset.palette.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit checked pairs per split; 0 checks all.")
    return parser.parse_args()


def image_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def match_label(image_path: Path, label_dir: Path) -> Optional[Path]:
    stems = [
        image_path.stem,
        image_path.stem.replace("_leftImg8bit", "_gtFine_labelIds"),
        image_path.stem.replace("_img", "_label"),
    ]
    for stem in stems:
        for ext in [".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"]:
            candidate = label_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    matches = sorted(label_dir.rglob(f"{image_path.stem}.*"))
    return matches[0] if matches else None


def find_split_root(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train", "training", "uavid_train"],
        "val": ["val", "validation", "uavid_val"],
        "test": ["test", "testing", "uavid_test"],
    }
    for name in aliases.get(split, [split]):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / split


def discover_pairs(root: Path, split: str) -> tuple[list[tuple[Path, Path]], list[str]]:
    split_root = find_split_root(root, split)
    issues: list[str] = []
    if not split_root.exists():
        return [], [f"Split root not found: {split_root}"]

    image_names = {name.lower() for name in DEFAULT_IMAGE_DIRS}
    label_names = {name.lower() for name in DEFAULT_LABEL_DIRS}
    image_dirs = [p for p in split_root.rglob("*") if p.is_dir() and p.name.lower() in image_names]
    label_dirs = {p.parent: p for p in split_root.rglob("*") if p.is_dir() and p.name.lower() in label_names}

    pairs: list[tuple[Path, Path]] = []
    for image_dir in image_dirs:
        label_dir = label_dirs.get(image_dir.parent)
        if label_dir is None:
            issues.append(f"No label directory next to image directory: {image_dir}")
            continue
        for image_path in sorted(image_files(image_dir)):
            label_path = match_label(image_path, label_dir)
            if label_path is None:
                issues.append(f"Missing label for image: {image_path}")
            else:
                pairs.append((image_path, label_path))

    if not pairs:
        image_dir = next((split_root / name for name in DEFAULT_IMAGE_DIRS if (split_root / name).exists()), None)
        label_dir = next((split_root / name for name in DEFAULT_LABEL_DIRS if (split_root / name).exists()), None)
        if image_dir and label_dir:
            for image_path in sorted(image_files(image_dir)):
                label_path = match_label(image_path, label_dir)
                if label_path is None:
                    issues.append(f"Missing label for image: {image_path}")
                else:
                    pairs.append((image_path, label_path))

    if not pairs and not issues:
        issues.append(f"No image/label pairs found under split: {split_root}")
    return pairs, issues


def load_palette(path: Optional[str]) -> Optional[Dict[Tuple[int, int, int], int]]:
    if not path:
        return None
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    palette = (payload.get("dataset") or {}).get("palette")
    if not palette:
        return None
    parsed: Dict[Tuple[int, int, int], int] = {}
    for key, value in palette.items():
        if isinstance(key, str):
            color = tuple(int(x.strip()) for x in key.replace("(", "").replace(")", "").split(","))
        else:
            color = tuple(int(x) for x in key)
        if len(color) != 3:
            raise ValueError(f"Palette key must have three channels: {key}")
        parsed[color] = int(value)
    return parsed


def read_label_values(path: Path, palette: Optional[Dict[Tuple[int, int, int], int]]) -> tuple[set[int], list[str]]:
    from PIL import Image

    issues: list[str] = []
    label = Image.open(path)
    if label.mode in {"RGB", "RGBA"}:
        rgb = label.convert("RGB")
        pixels = list(rgb.getdata())
        if palette:
            values = set()
            unknown = set()
            for pixel in pixels:
                if pixel in palette:
                    values.add(palette[pixel])
                else:
                    unknown.add(pixel)
                    if len(unknown) > 8:
                        break
            if unknown:
                issues.append(f"Unknown RGB label colors in {path}: {sorted(unknown)[:8]}")
            return values, issues
        if all(r == g == b for r, g, b in pixels):
            return {r for r, _, _ in pixels}, issues
        issues.append(f"RGB label requires palette: {path}")
        return set(), issues
    return set(label.convert("L").getdata()), issues


def check_pairs(
    split: str,
    pairs: list[tuple[Path, Path]],
    num_classes: int,
    ignore_index: int,
    palette: Optional[Dict[Tuple[int, int, int], int]],
    max_samples: int,
) -> list[str]:
    from PIL import Image

    issues: list[str] = []
    stems: dict[str, list[Path]] = defaultdict(list)
    for image_path, _ in pairs:
        stems[image_path.stem].append(image_path)
    for stem, paths in stems.items():
        if len(paths) > 1:
            issues.append(f"Duplicate image stem in split {split}: {stem}")

    checked_pairs = pairs[:max_samples] if max_samples > 0 else pairs
    valid_values = set(range(num_classes)) | {ignore_index}
    for image_path, label_path in checked_pairs:
        image = Image.open(image_path)
        label = Image.open(label_path)
        if image.size != label.size:
            issues.append(f"Size mismatch: {image_path} {image.size} vs {label_path} {label.size}")
        values, value_issues = read_label_values(label_path, palette)
        issues.extend(value_issues)
        invalid = sorted(v for v in values if v not in valid_values)
        if invalid:
            issues.append(f"Invalid label values in {label_path}: {invalid[:20]}")
    return issues


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    palette = load_palette(args.palette)
    all_issues: list[str] = []
    stems_by_split: dict[str, set[str]] = {}

    for split in args.splits:
        pairs, issues = discover_pairs(root, split)
        all_issues.extend(issues)
        if pairs:
            print(f"[PASS] {split}: found {len(pairs)} image/label pairs")
        split_issues = check_pairs(split, pairs, args.num_classes, args.ignore_index, palette, args.max_samples)
        all_issues.extend(split_issues)
        stems_by_split[split] = {image_path.stem for image_path, _ in pairs}

    for i, split_a in enumerate(args.splits):
        for split_b in args.splits[i + 1 :]:
            overlap = stems_by_split.get(split_a, set()) & stems_by_split.get(split_b, set())
            if overlap:
                all_issues.append(f"Potential split overlap between {split_a} and {split_b}: {sorted(overlap)[:20]}")

    if all_issues:
        for issue in all_issues:
            print(f"[FAIL] {issue}")
        return 1

    print("[PASS] File matching")
    print("[PASS] Image/label dimensions")
    print("[PASS] Label values")
    print("[PASS] Split overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
