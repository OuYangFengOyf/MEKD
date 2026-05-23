from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT


def _resize_image(img, size: Tuple[int, int]):
    return img.resize((size[1], size[0]), BILINEAR)


def _resize_mask(mask, size: Tuple[int, int]):
    return mask.resize((size[1], size[0]), NEAREST)


class SegTransform:
    def __init__(
        self,
        image_size: Tuple[int, int],
        train: bool,
        scale_range=(0.5, 2.0),
        hflip_prob: float = 0.5,
        color_jitter: float = 0.2,
    ):
        self.image_size = tuple(image_size)
        self.train = train
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

    def __call__(self, image, mask) -> Dict[str, torch.Tensor]:
        if self.train:
            scale = random.uniform(*self.scale_range)
            scaled = (max(1, int(image.height * scale)), max(1, int(image.width * scale)))
            image = image.resize((scaled[1], scaled[0]), BILINEAR)
            mask = mask.resize((scaled[1], scaled[0]), NEAREST)

            pad_h = max(0, self.image_size[0] - image.height)
            pad_w = max(0, self.image_size[1] - image.width)
            if pad_h or pad_w:
                image_t = pil_to_tensor(image).unsqueeze(0)
                mask_t = torch.as_tensor(list(mask.getdata()), dtype=torch.long).view(mask.height, mask.width)
                image_t = F.pad(image_t, (0, pad_w, 0, pad_h), value=0)
                mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), value=255)
                image = tensor_to_pil(image_t.squeeze(0))
                mask = tensor_to_mask_pil(mask_t)

            top = random.randint(0, image.height - self.image_size[0])
            left = random.randint(0, image.width - self.image_size[1])
            box = (left, top, left + self.image_size[1], top + self.image_size[0])
            image = image.crop(box)
            mask = mask.crop(box)

            if random.random() < self.hflip_prob:
                image = image.transpose(FLIP_LEFT_RIGHT)
                mask = mask.transpose(FLIP_LEFT_RIGHT)

            if self.color_jitter > 0:
                b = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
                c = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
                s = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
                image = ImageEnhance.Brightness(image).enhance(b)
                image = ImageEnhance.Contrast(image).enhance(c)
                image = ImageEnhance.Color(image).enhance(s)
        else:
            image = _resize_image(image, self.image_size)
            mask = _resize_mask(mask, self.image_size)

        return {"image": normalize(pil_to_tensor(image)), "mask": pil_mask_to_tensor(mask)}


def pil_to_tensor(image) -> torch.Tensor:
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.convert("RGB").tobytes()))
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return data


def tensor_to_pil(tensor: torch.Tensor):
    from PIL import Image

    tensor = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(tensor)


def tensor_to_mask_pil(mask: torch.Tensor):
    from PIL import Image

    return Image.fromarray(mask.byte().cpu().numpy())


def pil_mask_to_tensor(mask) -> torch.Tensor:
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    return data.view(mask.height, mask.width).long()


def normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std
