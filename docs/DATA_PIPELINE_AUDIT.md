# Data Pipeline Audit

## Dataset Structure

Supported UAVid-style layout:

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

Supported flat layout:

```text
dataset_root/
  train/images/*.png
  train/labels/*.png
  val/images/*.png
  val/labels/*.png
  test/images/*.png
  test/labels/*.png
```

The loader also accepts common folder aliases such as `Images`, `Labels`, `img`, `masks`, and `gt`.

## Pair Matching

- Image files are discovered recursively and sorted.
- Labels are matched by stem rather than by raw list position.
- Common suffix conversions such as `_leftImg8bit` to `_gtFine_labelIds` are supported.
- The dataset checker added in `tools/check_dataset_structure.py` verifies missing labels, duplicate stems, split overlap, image sizes, and label values.

## Label Handling

- Grayscale label-id masks are supported directly.
- RGB masks require `dataset.palette`, unless all RGB channels are identical and values are valid label IDs.
- Label resize uses nearest-neighbor interpolation.
- `ignore_index` is configurable and defaults to 255.

## Image Handling

- Images are read through PIL and converted to RGB.
- Values are converted to `[0, 1]`.
- ImageNet mean/std normalization is applied.
- Validation/test resize uses bilinear interpolation for images.

## Synchronized Augmentation

The following geometry transforms are synchronized between image and label:

- random scale
- pad
- crop
- horizontal flip
- validation/test resize

Color jitter is applied only to the image, which is appropriate for segmentation labels.

## Data Leakage

- The loader assumes separate split directories.
- It does not use random split generation.
- Duplicate stems across splits are checked by `tools/check_dataset_structure.py`.

Data leakage risk: Medium before running the checker on the real dataset, Low if the checker reports no split overlap.
