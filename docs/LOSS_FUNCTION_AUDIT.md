# Loss Function Audit

## Logits and Probabilities

| Output | Loss | Status |
| --- | --- | --- |
| Student raw multi-class logits | Cross entropy + Dice over softmax | Pass |
| Expert raw multi-class logits | Cross entropy | Pass |
| Density sigmoid probability | Balanced BCE | Pass |
| Student/teacher raw logits | Temperature KL distillation | Pass |
| Softmax probabilities for boundary maps | L1 edge distillation | Pass |

No repeated sigmoid or softmax-to-cross-entropy mismatch was found.

## Auxiliary Losses

- `Lproto`: prototype contrastive objective for Transformer features.
- `Lden`: density prediction loss for UAV-aware structure priors.
- `Llogit`: reliability-gated logit KD.
- `Lmsfd`: multi-scale feature distillation.
- `Lbd`: boundary-aware distillation.
- `Lroute`: router KL target from expert errors.

The losses are active in the expected stages:

- Warm-up: student segmentation plus expert losses.
- Distillation: student segmentation plus KD/routing losses.

## Numerical Stability

- Dice and BCE include epsilon/clamping.
- KD weights divide by clamped positive sums.
- Empty valid-label cases are handled in `prototype_loss`.
- Bad label IDs outside `[0, K-1]` can still break cross entropy; this is checked by `tools/check_dataset_structure.py`.

## Remaining Runtime Check

Loss/backward was not executed because PyTorch is unavailable in the current environment.
