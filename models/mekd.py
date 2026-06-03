from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from losses import (
    balanced_bce,
    boundary_distill_loss,
    logit_kd_loss,
    prototype_loss,
    segmentation_loss,
)
from priors import density_target, hard_region_mask
from .experts import MambaSpatialExpert, TransformerSemanticExpert
from .layers import ConvBNAct
from .stdc import STDCEncoder
from .student import STDCFPNStudent


class MEKDUAVSeg(nn.Module):
    def __init__(self, cfg: Dict, class_weights: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg["dataset"]["num_classes"])
        self.ignore_index = int(cfg["dataset"]["ignore_index"])
        self.class_weights = nn.Parameter(class_weights.float(), requires_grad=False)

        scfg = cfg["model"]["student"]
        ecfg = cfg["model"]["experts"]
        self.student = STDCFPNStudent(
            self.num_classes,
            base_channels=scfg["base_channels"],
            fpn_channels=scfg["fpn_channels"],
            dropout=scfg["dropout"],
        )
        ch = self.student.out_channels
        expert_ch = ecfg["channels"]
        self.transformer_semantic_expert = TransformerSemanticExpert(
            ch[2], ch[3], expert_ch, self.num_classes, ecfg["transformer_layers"], ecfg["transformer_heads"]
        )
        self.structure_density_predictor = nn.Sequential(ConvBNAct(ch[1], 64, 3), nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.mamba_spatial_expert = MambaSpatialExpert(ch[2], ch[3], expert_ch, self.num_classes, ecfg["mamba_blocks"])
        self.expert_router = nn.Sequential(
            ConvBNAct(expert_ch * 2 + 2, expert_ch, 1),
            ConvBNAct(expert_ch, expert_ch // 2, 3),
            nn.Conv2d(expert_ch // 2, 2, 1),
        )
        self.student_proj = nn.ModuleDict(
            {
                "1/8": ConvBNAct(ch[1], expert_ch, 1, act=False),
                "1/16": ConvBNAct(ch[2], expert_ch, 1, act=False),
                "1/32": ConvBNAct(ch[3], expert_ch, 1, act=False),
            }
        )
        self.teacher_side_encoder: Optional[STDCEncoder] = None
        self.register_buffer("prototypes", torch.zeros(self.num_classes, expert_ch))
        self.register_buffer("proto_seen", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("consistency_thr", torch.tensor(0.0))
        self.register_buffer("confidence_thr", torch.tensor(0.0))

    def init_teacher_from_student(self):
        self.teacher_side_encoder = copy.deepcopy(self.student.encoder)
        self.teacher_side_encoder.eval()
        for p in self.teacher_side_encoder.parameters():
            p.requires_grad = False
        for module in [self.transformer_semantic_expert, self.mamba_spatial_expert, self.structure_density_predictor]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def deployed_student(self) -> nn.Module:
        return self.student

    @torch.no_grad()
    def update_prototypes(self, feat: torch.Tensor, labels: torch.Tensor):
        lcfg = self.cfg["loss"]
        b, c, h, w = feat.shape
        labels_s = F.interpolate(labels.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
        feat_flat = feat.permute(0, 2, 3, 1)
        for cls_id in range(self.num_classes):
            mask = labels_s == cls_id
            count = int(mask.sum().item())
            if count <= int(lcfg["prototype_min_pixels"]):
                continue
            proto = feat_flat[mask].mean(dim=0)
            if not bool(self.proto_seen[cls_id]):
                self.prototypes[cls_id] = proto
                self.proto_seen[cls_id] = True
            else:
                eta = float(lcfg["prototype_momentum"])
                self.prototypes[cls_id] = eta * self.prototypes[cls_id] + (1.0 - eta) * proto

    def _source_features(self, images: torch.Tensor, student_feats, stage: str):
        if stage == "distill":
            if self.teacher_side_encoder is None:
                raise RuntimeError("Teacher encoder is not initialized. Call init_teacher_from_student() after warm-up.")
            with torch.no_grad():
                return self.teacher_side_encoder(images)
        return student_feats

    def forward(self, images: torch.Tensor, labels: torch.Tensor, stage: str, epoch_in_stage: int = 0, global_epoch: int = 0):
        lcfg = self.cfg["loss"]
        dcfg = self.cfg["dataset"]
        student_out = self.student(images)
        student_logits = student_out["logits"]
        src_feats = self._source_features(images, student_out["features"], stage)
        small_thr = int(dcfg.get("small_object_threshold", max(16, images.shape[-1] * images.shape[-2] // 2048)))

        density = self.structure_density_predictor(src_feats[1])
        t_feat, t_logits = self.transformer_semantic_expert(src_feats[2], src_feats[3])
        m_feat, m_logits = self.mamba_spatial_expert(src_feats[2], src_feats[3], density)
        t_logits_full = F.interpolate(t_logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        m_logits_full = F.interpolate(m_logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        loss_c = segmentation_loss(student_logits, labels, self.class_weights, lcfg["lambda_dice"], self.num_classes, self.ignore_index)
        if stage == "warmup":
            loss_t = segmentation_loss(t_logits_full, labels, self.class_weights, 0.0, self.num_classes, self.ignore_index)
            if global_epoch >= int(self.cfg["train"]["prototype_start_epoch"]) and bool(self.proto_seen.any()):
                loss_t = loss_t + lcfg["lambda_proto"] * prototype_loss(
                    t_feat, labels, self.prototypes, self.class_weights, lcfg["temperature_proto"], self.ignore_index
                )
            density_gt, _, _, _ = density_target(labels, self.num_classes, self.ignore_index, small_thr)
            density_gt_s = F.interpolate(density_gt, size=density.shape[-2:], mode="bilinear", align_corners=False)
            loss_den = balanced_bce(density, density_gt_s)
            loss_m = segmentation_loss(m_logits_full, labels, self.class_weights, 0.0, self.num_classes, self.ignore_index)
            loss_m = loss_m + lcfg["lambda_den"] * loss_den
            total = loss_c + lcfg["lambda_t"] * loss_t + lcfg["lambda_m"] * loss_m
            return {
                "loss": total,
                "loss_cseg": loss_c.detach(),
                "loss_tseg": loss_t.detach(),
                "loss_mseg": loss_m.detach(),
                "logits": student_logits.detach(),
                "t_feat": t_feat.detach(),
            }

        hard_mask = hard_region_mask(
            labels,
            student_logits,
            self.class_weights,
            self.num_classes,
            self.ignore_index,
            small_thr,
            epoch_in_stage >= int(self.cfg["train"]["uncertainty_start_epoch"]),
        )
        de = F.interpolate(density, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
        he = F.interpolate(hard_mask, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
        route_logits = self.expert_router(torch.cat([t_feat.detach(), m_feat.detach(), de.detach(), he.detach()], dim=1))
        alpha = torch.softmax(route_logits, dim=1)
        f_tea = alpha[:, 0:1] * t_feat.detach() + alpha[:, 1:2] * m_feat.detach()
        alpha_p = F.interpolate(alpha, size=t_logits.shape[-2:], mode="bilinear", align_corners=False)
        p_tea = alpha_p[:, 0:1] * t_logits.detach() + alpha_p[:, 1:2] * m_logits.detach()

        gate = self._reliability_gate(t_logits, m_logits)
        loss_logit = logit_kd_loss(
            student_logits, p_tea, labels, gate, hard_mask, self.class_weights, lcfg["temperature_logit"], lcfg["hard_region_beta"], self.ignore_index
        )
        loss_ms = self._multi_scale_feature_loss(student_out["features"], f_tea, labels, hard_mask)
        loss_bd = boundary_distill_loss(student_logits, p_tea)
        loss_route = self._routing_loss(alpha, t_logits, m_logits, labels)

        ramp = min(1.0, float(epoch_in_stage + 1) / max(1, int(self.cfg["train"]["distill_ramp_epochs"])))
        total = loss_c + ramp * (
            lcfg["lambda_ms"] * loss_ms
            + lcfg["lambda_logit"] * loss_logit
            + lcfg["lambda_bd"] * loss_bd
            + lcfg["lambda_route"] * loss_route
        )
        return {
            "loss": total,
            "loss_cseg": loss_c.detach(),
            "loss_logit": loss_logit.detach(),
            "loss_ms": loss_ms.detach(),
            "loss_bd": loss_bd.detach(),
            "loss_route": loss_route.detach(),
            "logits": student_logits.detach(),
        }

    def _reliability_gate(self, t_logits: torch.Tensor, m_logits: torch.Tensor) -> torch.Tensor:
        lcfg = self.cfg["loss"]
        qt = torch.softmax(t_logits, dim=1)
        qm = torch.softmax(m_logits, dim=1)
        mix = 0.5 * (qt + qm)
        js = 0.5 * (
            F.kl_div(mix.clamp_min(1e-6).log(), qt, reduction="none").sum(1, keepdim=True)
            + F.kl_div(mix.clamp_min(1e-6).log(), qm, reduction="none").sum(1, keepdim=True)
        )
        consistency = 1.0 - js
        confidence = torch.maximum(qt.max(1, keepdim=True).values, qm.max(1, keepdim=True).values)
        with torch.no_grad():
            q_c = torch.quantile(consistency.flatten(), 0.1)
            q_r = torch.quantile(confidence.flatten(), 0.1)
            m = float(lcfg["ema_threshold_momentum"])
            self.consistency_thr.mul_(m).add_(q_c * (1 - m))
            self.confidence_thr.mul_(m).add_(q_r * (1 - m))
        tau = float(lcfg["temperature_gate"])
        return torch.sigmoid((consistency - self.consistency_thr) / tau) * torch.sigmoid((confidence - self.confidence_thr) / tau)

    def _multi_scale_feature_loss(self, student_feats, f_tea, labels, hard_mask):
        losses = []
        for key, idx in [("1/8", 1), ("1/16", 2), ("1/32", 3)]:
            s = self.student_proj[key](student_feats[idx])
            t = F.interpolate(f_tea.detach(), size=s.shape[-2:], mode="bilinear", align_corners=False)
            hm = F.interpolate(hard_mask, size=s.shape[-2:], mode="bilinear", align_corners=False)
            lab = F.interpolate(labels.unsqueeze(1).float(), size=s.shape[-2:], mode="nearest").squeeze(1).long()
            valid = (lab != self.ignore_index).unsqueeze(1)
            cw = self.class_weights.to(s.device)[lab.clamp(0, self.num_classes - 1)].unsqueeze(1)
            weight = cw * hm * valid
            losses.append((((s - t) ** 2) * weight).sum() / weight.sum().clamp_min(1.0))
        return sum(losses) / len(losses)

    def _routing_loss(self, alpha, t_logits, m_logits, labels):
        lcfg = self.cfg["loss"]
        labels_s = F.interpolate(labels.unsqueeze(1).float(), size=t_logits.shape[-2:], mode="nearest").squeeze(1).long()
        et = F.cross_entropy(t_logits, labels_s, ignore_index=self.ignore_index, reduction="none")
        em = F.cross_entropy(m_logits, labels_s, ignore_index=self.ignore_index, reduction="none")
        target_t = torch.exp(-et / float(lcfg["temperature_route"]))
        target_m = torch.exp(-em / float(lcfg["temperature_route"]))
        target = torch.stack([target_t, target_m], dim=1)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        valid = (labels_s != self.ignore_index).unsqueeze(1)
        kl = F.kl_div(alpha.clamp_min(1e-6).log(), target.detach(), reduction="none").sum(dim=1, keepdim=True)
        return (kl * valid).sum() / valid.sum().clamp_min(1)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.student(images)["logits"]
