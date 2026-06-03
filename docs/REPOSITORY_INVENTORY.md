# Repository Inventory

## Root Files

- `README.md`
- `requirements.txt`
- `config.py`
- `utils.py`
- `train.py`
- `validate.py`
- `test.py`
- `benchmark.py`
- `export_student.py`
- `losses.py`
- `metrics.py`
- `priors.py`
- `mekd.pdf`
- `__init__.py`

## Source Code

- `models/`
  - `models/mekd.py`
  - `models/student.py`
  - `models/stdc.py`
  - `models/experts.py`
  - `models/layers.py`
  - `models/__init__.py`
- `data/`
  - `data/datasets.py`
  - `data/transforms.py`
  - `data/build.py`
  - `data/__init__.py`
- `configs/`
  - `configs/default.yaml`
  - `configs/example_dataset.yaml`
  - `configs/uavid_mekd.yaml`
  - `configs/udd6_mekd.yaml`
- `tools/`
  - `tools/smoke_test.py`
  - `tools/check_dataset_structure.py`
- `docs/`
  - audit documents generated during this review.

## Entry Points

- Training: `python train.py --config configs/uavid_mekd.yaml`
- Validation: `python validate.py --config ... --checkpoint ...`
- Testing: `python test.py --config ... --checkpoint ... --split test`
- Benchmark JSON: `python benchmark.py --config ... --checkpoint ... --output ...`
- Student export: `python export_student.py --config ... --checkpoint ... --output ...`
- Smoke test: `python tools/smoke_test.py`
- Dataset check: `python tools/check_dataset_structure.py --help`

## Paper Files

- `mekd.pdf`

## Missing but Expected Files

No blocking source files are missing after this audit. The repository still does not include raw datasets, checkpoints, or logs, which is acceptable for this scope.

## Notable Structure Findings

- No author-local absolute paths were found.
- `test.py`, `tools/smoke_test.py`, `tools/check_dataset_structure.py`, `configs/default.yaml`, and `configs/example_dataset.yaml` were added during the audit.
- The original README referenced a non-existent `mekd/` package layout; it has been revised.
