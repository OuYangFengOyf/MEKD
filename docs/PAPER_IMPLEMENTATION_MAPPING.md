# Paper Implementation Mapping

## Task

- Input: UAV RGB image `I` with shape `3 x H x W`.
- Output: dense semantic label map with `K` classes.
- Dataset: UAVid and UDD6, according to Section 4.1.1 of the local PDF.
- Metrics: mIoU, mF1, OA. The code also reports Precision and Recall after audit fixes.

## Architecture

- Backbone: lightweight STDC-style CNN encoder in the deployed student.
- Main modules:
  - STDC-FPN student.
  - Transformer semantic expert.
  - Mamba-style spatial expert.
  - Structure-density prior and hard-region mask.
  - Conflict-suppressed expert router.
  - Logit, multi-scale feature, and boundary distillation.
- Decoder: lightweight FPN decoder with 1x1 projections, depthwise separable convolutions, bilinear upsampling, and skip fusion.
- Output head: multi-class segmentation logits with shape `B x K x H x W`.

## Data Flow

- Input image shape: configured by `dataset.image_size`, default `[512, 1024]`.
- Multi-stage features: `1/4`, `1/8`, `1/16`, `1/32`.
- Feature interaction:
  - Experts consume `[F(1/16), Up(F(1/32))]`.
  - Router consumes Transformer feature, Mamba feature, density map, and hard-region mask.
- Fusion: spatially adaptive expert routing produces fused teacher logits/features.
- Prediction: deployed inference uses only `student.encoder + student.decoder`.

## Loss Functions

- Main loss: class-balanced cross entropy plus Dice loss for the student.
- Auxiliary losses:
  - Transformer expert segmentation loss.
  - Prototype loss after prototype warm-up.
  - Mamba expert segmentation loss.
  - Balanced BCE density supervision.
  - Logit KD.
  - Multi-scale feature distillation.
  - Boundary distillation.
  - Routing KL loss.
- Loss weights: configured in `config.py` and YAML overrides.

## Training Settings

- Optimizer: AdamW.
- Learning rate: `4e-4` in the provided configs.
- Scheduler: warm-up plus cosine annealing implemented with `LambdaLR`.
- Epochs: Not explicitly specified in the revised paper.
- Batch size: 16.
- Input size: `[512, 1024]`.
- Data augmentation: random scale, random crop/pad, horizontal flip, color jitter, ImageNet normalization.
- Random seed: 42 in provided configs.

## Evaluation

- Threshold: not applicable; this is multi-class segmentation and uses `argmax`.
- Metrics: Precision, Recall, IoU, F1, OA, plus mean variants.
- Validation protocol: resize-based evaluation to configured `dataset.image_size`.
- Test protocol: `test.py` wraps the same evaluation path and can override `--split test`.

## Paper-to-Code Mapping

| Paper component | Expected code file | Actual code file | Status |
|---|---|---|---|
| UAV semantic segmentation task | data loader and training entry | `data/datasets.py`, `train.py` | Pass |
| STDC2-FPN student | student model | `models/student.py`, `models/stdc.py`, `models/layers.py` | Pass |
| Alternative student backbones | model factory | Not implemented | Warning |
| Transformer semantic expert | expert module | `models/experts.py` | Pass |
| Mamba spatial expert | expert module | `models/experts.py` | Pass, simplified Mamba-style scan |
| Structure density prior | prior module | `priors.py` | Pass |
| Hard-region prior | prior module | `priors.py` | Pass |
| Conflict-suppressed router | fusion module | `models/mekd.py` | Pass |
| Class-balanced CE + Dice | loss module | `losses.py` | Pass |
| Prototype objective | loss/module | `losses.py`, `models/mekd.py` | Pass |
| Logit KD | loss module | `losses.py` | Pass |
| Multi-scale feature KD | model loss | `models/mekd.py` | Pass |
| Boundary KD | loss module | `losses.py` | Pass |
| Two-stage warm-up/distillation | training loop | `train.py`, `models/mekd.py` | Pass |
| Optional self-distillation fine-tuning | training stage | Not implemented | Warning |
| UAVid and UDD6 metrics | metrics module | `metrics.py` | Pass |
| Sliding-window inference | evaluation | Not implemented | Warning |
| FPS/GFLOPs/latency efficiency evaluation | benchmark script | Not implemented | Warning |

## Paper File Integrity

Rechecked after PDF revision: the local PDF now consistently describes MEKD-UAVSeg for UAV semantic segmentation. No C2M-Net/autonomous-driving text was found by text search.
