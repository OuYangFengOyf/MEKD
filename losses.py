from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .priors import one_hot, sobel_edge


def class_balanced_weights(freq: torch.Tensor, mu: float) -> torch.Tensor:
    return 1.0 / torch.log(mu + freq.clamp_min(1e-8).float())


def balanced_ce(logits, target, class_weights, ignore_index=255):
    return F.cross_entropy(logits, target, weight=class_weights.to(logits.device), ignore_index=ignore_index)


def dice_loss(logits, target, num_classes, ignore_index=255, eps=1e-6):
    prob = torch.softmax(logits, dim=1)
    oh = one_hot(target, num_classes, ignore_index)
    valid = (target != ignore_index).unsqueeze(1)
    prob = prob * valid
    inter = (prob * oh).sum(dim=(0, 2, 3))
    den = prob.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (den + eps)).mean()


def segmentation_loss(logits, target, class_weights, lambda_dice, num_classes, ignore_index):
    return balanced_ce(logits, target, class_weights, ignore_index) + lambda_dice * dice_loss(logits, target, num_classes, ignore_index)


def balanced_bce(pred, target, eps=1e-6):
    pos = target.mean().clamp(eps, 1 - eps)
    weight = target * (1 - pos) + (1 - target) * pos
    return F.binary_cross_entropy(pred.clamp(eps, 1 - eps), target, weight=weight)


def prototype_loss(feat, labels, prototypes, class_weights, temperature, ignore_index):
    b, c, h, w = feat.shape
    labels_s = F.interpolate(labels.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
    valid = (labels_s != ignore_index) & (labels_s >= 0) & (labels_s < prototypes.shape[0])
    if valid.sum() == 0:
        return feat.sum() * 0.0
    f = F.normalize(feat.permute(0, 2, 3, 1)[valid], dim=1)
    p = F.normalize(prototypes.to(feat.device), dim=1)
    logits = f @ p.t() / temperature
    target = labels_s[valid]
    loss = F.cross_entropy(logits, target, weight=class_weights.to(feat.device), reduction="none")
    return loss.mean()


def logit_kd_loss(student_logits, teacher_logits, target, gate, hard_mask, class_weights, temperature, beta, ignore_index):
    t = temperature
    teacher = F.interpolate(teacher_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
    gate = F.interpolate(gate, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
    hard_mask = F.interpolate(hard_mask, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
    valid = (target != ignore_index).unsqueeze(1)
    pixel_w = class_weights.to(student_logits.device)[target.clamp(0, len(class_weights) - 1)].unsqueeze(1)
    weight = pixel_w * gate * (1.0 + beta * hard_mask) * valid
    kd = F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher.detach() / t, dim=1),
        reduction="none",
    ).sum(dim=1, keepdim=True)
    return (kd * weight).sum() * (t * t) / weight.sum().clamp_min(1.0)


def boundary_distill_loss(student_logits, teacher_logits):
    teacher = F.interpolate(teacher_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
    se = sobel_edge(torch.softmax(student_logits, dim=1))
    te = sobel_edge(torch.softmax(teacher.detach(), dim=1))
    return F.l1_loss(se, te)
