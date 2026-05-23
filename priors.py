from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid = (labels >= 0) & (labels < num_classes) & (labels != ignore_index)
    safe = labels.clamp(0, num_classes - 1)
    oh = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).float()
    return oh * valid.unsqueeze(1)


def sobel_edge(prob_or_onehot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    c = prob_or_onehot.shape[1]
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=prob_or_onehot.dtype, device=prob_or_onehot.device)
    sy = sx.t()
    sx = sx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    sy = sy.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(prob_or_onehot, sx, padding=1, groups=c)
    gy = F.conv2d(prob_or_onehot, sy, padding=1, groups=c)
    edge = torch.sqrt(gx.pow(2) + gy.pow(2) + eps).sum(dim=1, keepdim=True)
    return edge


def percentile_norm(x: torch.Tensor, q_low=0.02, q_high=0.98, eps=1e-6) -> torch.Tensor:
    b = x.shape[0]
    flat = x.flatten(1)
    lo = torch.quantile(flat, q_low, dim=1).view(b, 1, 1, 1)
    hi = torch.quantile(flat, q_high, dim=1).view(b, 1, 1, 1)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1)


def boundary_mask(labels: torch.Tensor, num_classes: int, ignore_index: int, dilate: int = 3) -> torch.Tensor:
    edge = sobel_edge(one_hot(labels, num_classes, ignore_index))
    edge = (edge > 0).float()
    if dilate > 1:
        edge = F.max_pool2d(edge, kernel_size=dilate, stride=1, padding=dilate // 2)
    return edge


def small_object_mask(labels: torch.Tensor, num_classes: int, ignore_index: int, threshold: int) -> torch.Tensor:
    masks = []
    for label in labels.detach().cpu():
        h, w = label.shape
        out = torch.zeros((h, w), dtype=torch.float32)
        visited = torch.zeros((h, w), dtype=torch.bool)
        for y in range(h):
            for x in range(w):
                cls = int(label[y, x])
                if visited[y, x] or cls == ignore_index or cls < 0 or cls >= num_classes:
                    continue
                coords = []
                q = deque([(y, x)])
                visited[y, x] = True
                while q:
                    cy, cx = q.popleft()
                    coords.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and int(label[ny, nx]) == cls:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                if len(coords) < threshold:
                    for cy, cx in coords:
                        out[cy, cx] = 1.0
        masks.append(out)
    return torch.stack(masks, dim=0).unsqueeze(1).to(labels.device)


def local_entropy(labels: torch.Tensor, num_classes: int, ignore_index: int, windows=(3, 7, 15)) -> torch.Tensor:
    oh = one_hot(labels, num_classes, ignore_index)
    ents = []
    for k in windows:
        p = F.avg_pool2d(oh, kernel_size=k, stride=1, padding=k // 2).clamp_min(1e-6)
        ents.append(-(p * p.log()).sum(dim=1, keepdim=True))
    return sum(ents) / len(ents)


def density_target(labels: torch.Tensor, num_classes: int, ignore_index: int, small_threshold: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = percentile_norm(boundary_mask(labels, num_classes, ignore_index))
    s = percentile_norm(small_object_mask(labels, num_classes, ignore_index, small_threshold))
    c = percentile_norm(local_entropy(labels, num_classes, ignore_index))
    density = percentile_norm(1.0 - (1.0 - b) * (1.0 - s) * (1.0 - c))
    return density, b, s, c


def hard_region_mask(
    labels: torch.Tensor,
    student_logits: torch.Tensor,
    class_weights: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    small_threshold: int,
    uncertainty_enabled: bool,
) -> torch.Tensor:
    _, b, s, _ = density_target(labels, num_classes, ignore_index, small_threshold)
    rare = class_weights.to(labels.device)[labels.clamp(0, num_classes - 1)].unsqueeze(1)
    rare = rare * ((labels != ignore_index).unsqueeze(1))
    rare = percentile_norm(rare)
    if uncertainty_enabled:
        prob = torch.softmax(student_logits.detach(), dim=1)
        uncertainty = percentile_norm(1.0 - prob.max(dim=1, keepdim=True).values)
    else:
        uncertainty = torch.zeros_like(b)
    return percentile_norm(1.0 - (1.0 - s) * (1.0 - b) * (1.0 - rare) * (1.0 - uncertainty))
