# Model Architecture Audit

## Input and Output

Input:

```text
B x 3 x H x W
```

Output:

```text
B x K x H x W
```

This is a multi-class segmentation model. It uses argmax predictions rather than binary thresholding.

## Siamese Encoder

Not applicable. This repository implements single-image UAV semantic segmentation, not bi-temporal change detection.

## Student Encoder

`STDCEncoder` returns:

| Stage | Operation | Output Shape |
| --- | --- | --- |
| input | RGB image | `B x 3 x H x W` |
| stem | stride 4 CNN stem | `B x 2C x ~H/4 x ~W/4` |
| stage8 | STDC bottlenecks | `B x 4C x ~H/8 x ~W/8` |
| stage16 | STDC bottlenecks | `B x 8C x ~H/16 x ~W/16` |
| stage32 | STDC bottlenecks | `B x 16C x ~H/32 x ~W/32` |

The decoder upsamples by explicit target shape, so non-32-multiple sizes should be structurally supported.

## Feature Interaction

| Component | Input | Operation | Output | Status |
| --- | --- | --- | --- | --- |
| Transformer expert | `F16`, `F32` | upsample `F32`, concat, project, Transformer encoder | expert feature and logits at `1/16` | Pass |
| Mamba-style expert | `F16`, `F32`, density | local DSConv plus four directional scans | expert feature and logits at `1/16` | Pass |
| Density predictor | `F8` | Conv, sigmoid | `B x 1 x ~H/8 x ~W/8` | Pass |
| Router | expert features, density, hard mask | softmax routing | two expert weights | Pass |
| Student FPN | `F4` to `F32` | top-down fusion | logits at input size | Pass |

## Decoder

The deployed decoder is a lightweight FPN in `models/layers.py`:

- 1x1 projections.
- Bilinear upsampling.
- Additive skip fusion.
- Depthwise separable smoothing.
- 1x1 class head.

## Runtime Shape Check

Not executed in this environment because PyTorch is unavailable. `tools/smoke_test.py` was added to perform this check when PyTorch is installed.
