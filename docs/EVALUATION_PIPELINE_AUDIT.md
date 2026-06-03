# Evaluation Pipeline Audit

## Evaluation Mode

- `model.eval()` is used.
- `torch.no_grad()` is used.
- Batch size is configurable through `train.batch_size` or optional `train.eval_batch_size`.
- Validation/test images are resized to `dataset.image_size`.
- No sliding-window inference is implemented.
- No original-size prediction mask saving is implemented.

## Metrics

The code accumulates a global confusion matrix and then computes:

- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1: `2TP / (2TP + FP + FN)`
- IoU: `TP / (TP + FP + FN)`
- OA: correct valid pixels over all valid pixels

For UDD6, mean metrics use classes `1-5` and exclude class `0`.

## Output Saving

- `validate.py` and `test.py` can save metrics JSON through `--output-json`.
- `benchmark.py` saves metrics and protocol metadata.
- Prediction mask image saving is not implemented.

## Fixes Made

- Added `test.py` as a test entry.
- Added `--split` override to `validate.py` and `test.py`.
- Evaluation now builds only the requested eval split and does not require a training split.
- Added Precision and Recall outputs.
