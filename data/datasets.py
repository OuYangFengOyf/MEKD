from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import SegTransform


UAVID_CLASSES = [
    "building",
    "road",
    "tree",
    "low_vegetation",
    "moving_car",
    "static_car",
    "human",
    "clutter",
]

UDD6_CLASSES = ["other", "facade", "road", "vegetation", "vehicle", "roof"]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


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
        for ext in [".png", ".bmp", ".tif", ".tiff", ".jpg"]:
            candidate = label_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    matches = sorted(label_dir.rglob(f"{image_path.stem}.*"))
    return matches[0] if matches else None


def split_root_with_aliases(root: Path, split: str) -> Path:
    split_root = root / split
    if split_root.exists():
        return split_root
    aliases = {
        "train": ["uavid_train", "training"],
        "val": ["uavid_val", "validation"],
        "test": ["uavid_test", "testing"],
    }
    for alias in aliases.get(split, []):
        candidate = root / alias
        if candidate.exists():
            return candidate
    return split_root


def discover_uavseg_samples(root: str | Path, split: str) -> List[Tuple[Path, Path]]:
    split_root = split_root_with_aliases(Path(root), split)
    pairs: List[Tuple[Path, Path]] = []
    image_names = {"images", "image", "imgs", "img"}
    label_names = {"labels", "label", "masks", "mask", "gt", "ann"}
    image_dirs = [p for p in split_root.rglob("*") if p.is_dir() and p.name.lower() in image_names]
    label_dirs = {p.parent: p for p in split_root.rglob("*") if p.is_dir() and p.name.lower() in label_names}

    for image_dir in image_dirs:
        label_dir = label_dirs.get(image_dir.parent)
        if label_dir is None:
            continue
        for image_path in sorted(image_files(image_dir)):
            label_path = match_label(image_path, label_dir)
            if label_path is not None:
                pairs.append((image_path, label_path))

    if not pairs:
        image_dir = next((split_root / n for n in ["images", "Images", "img"] if (split_root / n).exists()), None)
        label_dir = next((split_root / n for n in ["labels", "Labels", "masks", "Masks", "gt"] if (split_root / n).exists()), None)
        if image_dir and label_dir:
            for image_path in sorted(image_files(image_dir)):
                label_path = match_label(image_path, label_dir)
                if label_path is not None:
                    pairs.append((image_path, label_path))
    return pairs


class UAVSegDataset(Dataset):
    """Robust UAVid/UDD6 segmentation reader.

    It accepts the common UAVid layout with sequence folders
    (split/seq*/Images and split/seq*/Labels) and the common flat layout
    (split/images and split/labels). Labels can be grayscale ids or RGB masks
    converted by a user-provided palette.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        dataset_name: str,
        image_size: Sequence[int],
        num_classes: int,
        ignore_index: int = 255,
        train: bool = True,
        label_mode: str = "auto",
        palette: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.dataset_name = dataset_name.lower()
        self.image_size = tuple(int(x) for x in image_size)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_mode = label_mode
        self.palette = _parse_palette(palette)
        self.transform = SegTransform(self.image_size, train=train, ignore_index=ignore_index)
        self.samples = discover_uavseg_samples(self.root, self.split)
        if not self.samples:
            raise FileNotFoundError(
                f"No image/label pairs found for split '{split}' under {self.root}. "
                "Expected UAVid seq*/Images+Labels or flat images+labels folders."
            )

    @property
    def classes(self) -> List[str]:
        if self.dataset_name == "udd6":
            return UDD6_CLASSES
        return UAVID_CLASSES

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        label = self._read_label(label_path)
        out = self.transform(image, label)
        out["path"] = str(image_path)
        return out

    def _read_label(self, path: Path):
        label = Image.open(path)
        if self.label_mode == "rgb" or (self.label_mode == "auto" and label.mode in {"RGB", "RGBA"}):
            rgb = label.convert("RGB")
            data = torch.ByteTensor(torch.ByteStorage.from_buffer(rgb.tobytes())).view(rgb.height, rgb.width, 3)
            if not self.palette:
                channels_equal = (data[..., 0] == data[..., 1]).all() and (data[..., 0] == data[..., 2]).all()
                gray = data[..., 0]
                values = torch.unique(gray)
                valid_values = ((values < self.num_classes) | (values == self.ignore_index)).all()
                if channels_equal and bool(valid_values):
                    from PIL import Image as PILImage

                    return PILImage.fromarray(gray.numpy())
                raise ValueError(
                    f"RGB label mask requires a palette: {path}. "
                    "Provide dataset.palette in the YAML config, or convert labels to grayscale class IDs."
                )
            mask = torch.full((rgb.height, rgb.width), self.ignore_index, dtype=torch.uint8)
            for color, cls_id in self.palette.items():
                color_t = torch.tensor(color, dtype=torch.uint8)
                hit = (data == color_t).all(dim=-1)
                mask[hit] = int(cls_id)
            from PIL import Image as PILImage

            return PILImage.fromarray(mask.numpy())
        return label.convert("L")

    def class_frequencies(self) -> torch.Tensor:
        counts = torch.zeros(self.num_classes, dtype=torch.float64)
        for _, label_path in self.samples:
            label = torch.as_tensor(list(self._read_label(label_path).getdata()), dtype=torch.long)
            valid = (label >= 0) & (label < self.num_classes)
            counts += torch.bincount(label[valid], minlength=self.num_classes).double()
        return counts / counts.sum().clamp_min(1)

    def connected_component_area_quantile(self, q: float) -> int:
        areas: List[int] = []
        for _, label_path in self.samples:
            label_img = self._read_label(label_path)
            label = torch.as_tensor(list(label_img.getdata()), dtype=torch.long).view(label_img.height, label_img.width)
            h, w = label.shape
            visited = torch.zeros((h, w), dtype=torch.bool)
            for y in range(h):
                for x in range(w):
                    cls = int(label[y, x])
                    if visited[y, x] or cls == self.ignore_index or cls < 0 or cls >= self.num_classes:
                        continue
                    area = 0
                    queue = deque([(y, x)])
                    visited[y, x] = True
                    while queue:
                        cy, cx = queue.popleft()
                        area += 1
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and int(label[ny, nx]) == cls:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                    areas.append(area)
        if not areas:
            return 16
        return max(1, int(torch.quantile(torch.tensor(areas, dtype=torch.float32), float(q)).item()))


def _parse_palette(palette) -> Optional[Dict[Tuple[int, int, int], int]]:
    if not palette:
        return None
    parsed: Dict[Tuple[int, int, int], int] = {}
    for key, value in palette.items():
        if isinstance(key, str):
            color = tuple(int(x) for x in key.replace("(", "").replace(")", "").split(","))
        else:
            color = tuple(int(x) for x in key)
        if len(color) != 3:
            raise ValueError(f"Palette color must have 3 channels, got {key}")
        parsed[color] = int(value)
    return parsed
