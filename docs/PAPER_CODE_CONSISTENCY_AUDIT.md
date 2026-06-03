# Paper-Code Consistency Audit

| Item | Paper Description | Code Implementation | Status | Required Fix |
| --- | --- | --- | --- | --- |
| Input format | UAV RGB image | PIL RGB image to normalized tensor | Pass | None |
| Output format | Dense semantic map | Multi-class logits `B x K x H x W` | Pass | None |
| Backbone | Lightweight CNN student, STDC2-FPN in experiments | STDC-style encoder plus FPN decoder | Pass | None |
| Alternative backbones | MobileNetV3, EfficientNet-Lite, ResNet18 can instantiate student | Not provided | Warning | Needs manual confirmation if required |
| Transformer expert | High-level semantic expert | `TransformerSemanticExpert` | Pass | None |
| Mamba expert | Directional spatial expert | Simplified four-direction selective scan | Warning | Needs manual confirmation against intended Mamba implementation |
| Decoder | Lightweight FPN | `FPNFuse` | Pass | None |
| Output head | Class logits | 1x1 conv heads | Pass | None |
| Main loss | Class-balanced CE + Dice | `segmentation_loss` | Pass | None |
| Auxiliary loss | Expert segmentation, density, prototype, KD, routing | Implemented across `losses.py` and `models/mekd.py` | Pass | None |
| Loss weights | Lambda weights in paper equations | Configurable in `config.py` | Pass | None |
| Data augmentation | Cropping/resizing and common segmentation transforms | Random scale, crop/pad, hflip, color jitter | Pass | None |
| Optimizer | AdamW | AdamW | Pass | None |
| Scheduler | Warm-up + cosine | LambdaLR warm-up + cosine | Pass | None |
| Epochs | Not explicitly specified in the revised paper | YAML uses 160 epochs | Warning | Document config as implementation choice |
| Evaluation threshold | Not applicable for multi-class | Argmax | Pass | None |
| Metrics | mIoU, mF1, OA | mPrecision, mRecall, mIoU, mF1, OA | Pass | None |
| UDD6 mean classes | Foreground classes only | classes 1-5 | Pass | None |
| Sliding-window test | Mentioned as possible protocol | Not implemented | Warning | Documented limitation |
| Self-distillation fine-tuning | Optional stage in paper | Not implemented | Warning | Needs manual confirmation |
| Efficiency metrics | FPS, GFLOPs, latency, parameters | Parameter count can be inferred; FPS/GFLOPs/latency script not implemented | Warning | Add efficiency benchmark if reproducing Table 4 |

## Main Finding

The revised PDF now consistently describes MEKD-UAVSeg, and the code implements the core training-time distillation framework and deployed student path. Remaining inconsistencies are implementation-scope limitations rather than obvious paper-file corruption.
