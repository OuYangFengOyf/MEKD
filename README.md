# MEKD-UAVSeg

**Conflict-Suppressed Heterogeneous Expert Distillation for Lightweight UAV Semantic Segmentation**

This folder contains the reproducible PyTorch code for MEKD-UAVSeg on UAVid and UDD6. It includes configs, fixed seeds, preprocessing, dataset split manifests, training, validation, benchmark logging, demo inference, and deployable-student export.



## 1. Install

```bash
conda create -n mekd python=3.9 -y
conda activate mekd
pip install -r mekd/requirements.txt
```

Main dependencies:

```text
torch==1.11.0
Pillow>=9.0.0
PyYAML>=6.0
tqdm>=4.64.0
```

CUDA is recommended for training.

## 2. Prepare Data

Raw UAV imagery is not redistributed in this repository. Users should download the datasets from official sources and follow the corresponding license or access terms.

### UAVid

Official source: <https://uavid.nl/>

UAVid is a high-resolution UAV semantic segmentation dataset for urban scenes. The official website describes the benchmark and provides access through the official UAVid/EOStore download path.

Expected label ids:

| ID | Class |
| --- | --- |
| 0 | building |
| 1 | road |
| 2 | tree |
| 3 | low_vegetation |
| 4 | moving_car |
| 5 | static_car |
| 6 | human |
| 7 | clutter |

Expected layout:

```text
data/UAVid/
  train/
    seq*/
      Images/*.png
      Labels/*.png
  val/
    seq*/
      Images/*.png
      Labels/*.png
```

The loader also accepts common split aliases such as `uavid_train`, `uavid_val`, `training`, and `validation`.

### UDD6

Official source: <https://github.com/MarcWong/UDD>

UDD is the Urban Drone Dataset for aerial semantic segmentation. The official GitHub page lists UDD-5 and UDD-6 download information and states the non-commercial use restriction.

Official UDD6 label-id order used by this code:

| ID | Class | RGB |
| --- | --- | --- |
| 0 | other | `(0, 0, 0)` |
| 1 | facade | `(102, 102, 156)` |
| 2 | road | `(128, 64, 128)` |
| 3 | vegetation | `(107, 142, 35)` |
| 4 | vehicle | `(0, 0, 142)` |
| 5 | roof | `(70, 70, 70)` |

Expected layout:

```text
data/UDD6/
  train/
    images/*.png
    labels/*.png
  val/
    images/*.png
    labels/*.png
```



## 3. Configs and Preprocessing

Main configs:

```text
mekd/configs/uavid_mekd.yaml
mekd/configs/udd6_mekd.yaml
```

Both configs use `seed: 42`. The trainer seeds Python `random`, PyTorch CPU, and PyTorch CUDA, then saves the merged runtime config to `config.resolved.json`.

Preprocessing is implemented in `mekd/data/transforms.py`:

| Stage | Preprocessing |
| --- | --- |
| Training | random scale `[0.5, 2.0]`, crop/pad to `512 x 1024`, horizontal flip `0.5`, color jitter `0.2`, ImageNet normalization |
| Validation | resize to `512 x 1024`, bilinear for images, nearest for labels, ImageNet normalization |

Grayscale label-id masks work directly. For RGB masks, set `dataset.palette` in the YAML config.

## 4. Train

```bash
python -m mekd.train --config mekd/configs/uavid_mekd.yaml
python -m mekd.train --config mekd/configs/udd6_mekd.yaml
```

Resume:

```bash
python -m mekd.train \
  --config mekd/configs/uavid_mekd.yaml \
  --resume runs/mekd_uavid/last.pth
```

Outputs:

```text
runs/mekd_uavid/config.resolved.json
runs/mekd_uavid/last.pth
runs/mekd_uavid/best.pth

runs/mekd_udd6/config.resolved.json
runs/mekd_udd6/last.pth
runs/mekd_udd6/best.pth
```

## 5. Validate

Validate:

```bash
python -m mekd.validate \
  --config mekd/configs/uavid_mekd.yaml \
  --checkpoint runs/mekd_uavid/best.pth \
  --output-json runs/mekd_uavid/val_metrics.json

python -m mekd.validate \
  --config mekd/configs/udd6_mekd.yaml \
  --checkpoint runs/mekd_udd6/best.pth \
  --output-json runs/mekd_udd6/val_metrics.json
```

Export the deployable student:

```bash
python -m mekd.export_student \
  --config mekd/configs/uavid_mekd.yaml \
  --checkpoint runs/mekd_uavid/best.pth \
  --output checkpoints/mekd_uavid_student_only.pth
```


Metrics include `mIoU`, `mF1`, `OA`, per-class IoU, and per-class F1. For UDD6, foreground classes `1-5` are averaged and class `0` (`other`) is excluded from mean metrics.


