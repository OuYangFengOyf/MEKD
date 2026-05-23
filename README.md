# MEKD-UAVSeg

PyTorch code for **MEKD-UAVSeg: Conflict-Suppressed Heterogeneous Expert Distillation for Lightweight UAV Semantic Segmentation**.

## Structure

```text
mekd/
  train.py              # two-stage warm-up + conflict-suppressed distillation
  validate.py           # validation with the deployed student
  export_student.py     # export STDC2-FPN student only
  configs/              # UAVid and UDD6 configs
  data/                 # UAVid/UDD6 loading and augmentation
  models/               # STDC2-FPN student, Transformer expert, Mamba expert, MEKD wrapper
  priors.py             # UAV-aware structure and hard-region priors
  losses.py             # segmentation, prototype, logit, feature, boundary KD losses
  metrics.py            # IoU, mIoU, mF1, OA
```

## Dataset

Supported layouts:

```text
data/UAVid/train/seq*/Images/*.png
data/UAVid/train/seq*/Labels/*.png
data/UAVid/val/seq*/Images/*.png
data/UAVid/val/seq*/Labels/*.png

data/UDD6/train/images/*.png
data/UDD6/train/labels/*.png
data/UDD6/val/images/*.png
data/UDD6/val/labels/*.png
```

Grayscale label-id masks work directly. For RGB masks, fill `dataset.palette` in the config as `"R,G,B": class_id`.

## Usage

```bash
pip install -r mekd/requirements.txt

python -m mekd.train --config mekd/configs/uavid_mekd.yaml
python -m mekd.train --config mekd/configs/udd6_mekd.yaml

python -m mekd.validate --config mekd/configs/uavid_mekd.yaml --checkpoint runs/mekd_uavid/best.pth

python -m mekd.export_student \
  --config mekd/configs/uavid_mekd.yaml \
  --checkpoint runs/mekd_uavid/best.pth \
  --output runs/mekd_uavid/student_only.pth
```

At inference, only the lightweight `STDCFPNStudent` is retained. The Transformer semantic expert, Mamba spatial expert, teacher-side encoder, structure-density predictor, expert router, auxiliary heads, and distillation adapters are training-only.
