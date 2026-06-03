# Configuration Audit

## Configuration Sources

Priority:

```text
script arguments for entry files
YAML config
defaults in config.py
```

The CLI selects config/checkpoint/resume/output paths. Algorithmic hyperparameters are controlled through YAML and `config.py` defaults.

## Configurable Fields

- Dataset root.
- Train/val/test split name.
- Number of classes.
- Ignore index.
- Palette for RGB label masks.
- Image size.
- Device.
- Seed.
- Batch size.
- Workers.
- Learning rate.
- Weight decay.
- Epochs.
- Warm-up epochs.
- Loss weights.
- Save directory.
- AMP.
- Deterministic cuDNN flag.

## Files

- `configs/default.yaml`
- `configs/uavid_mekd.yaml`
- `configs/udd6_mekd.yaml`
- `configs/example_dataset.yaml`

## Findings

- No hard-coded absolute local paths were found.
- The original configs were dataset-specific only; default/example configs were added.
- UDD6 RGB palette was added to `configs/udd6_mekd.yaml`.
- `requirements.txt` now uses a less rigid PyTorch requirement and includes NumPy.

## Remaining Notes

- Full command-line overrides for every YAML value are not implemented.
- The revised paper does not explicitly specify the total training epoch count; the YAML configs use 160 epochs.
