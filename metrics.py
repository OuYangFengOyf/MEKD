from __future__ import annotations

from typing import Iterable, Optional

import torch


class SegMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255, eval_classes: Optional[Iterable[int]] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eval_classes = list(eval_classes) if eval_classes is not None else list(range(num_classes))
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        pred = logits.argmax(dim=1).detach().cpu()
        target = target.detach().cpu()
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[valid].long() + pred[valid].long()
        self.confusion += torch.bincount(inds, minlength=self.num_classes ** 2).view(self.num_classes, self.num_classes).double()

    def compute(self):
        tp = self.confusion.diag()
        fp = self.confusion.sum(0) - tp
        fn = self.confusion.sum(1) - tp
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        iou = tp / (tp + fp + fn).clamp_min(1)
        f1 = (2 * tp) / (2 * tp + fp + fn).clamp_min(1)
        oa = tp.sum() / self.confusion.sum().clamp_min(1)
        cls = torch.tensor(self.eval_classes, dtype=torch.long)
        return {
            "mPrecision": precision[cls].mean().item() * 100,
            "mRecall": recall[cls].mean().item() * 100,
            "mIoU": iou[cls].mean().item() * 100,
            "mF1": f1[cls].mean().item() * 100,
            "OA": oa.item() * 100,
            "Precision": (precision * 100).tolist(),
            "Recall": (recall * 100).tolist(),
            "IoU": (iou * 100).tolist(),
            "F1": (f1 * 100).tolist(),
        }
