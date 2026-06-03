# Reproducibility Changelog

| File | Original Issue | Modification | Reason | Status |
| --- | --- | --- | --- | --- |
| `train.py` | Imported non-existent `mekd.*` package and required PyTorch for `--help` | Reworked imports to load runtime after CLI parsing | Make entry usable from repository root | Done |
| `validate.py` | Required training split during evaluation | Added eval-only loader and uniform bootstrap weights | Allow validation/test with only eval split plus checkpoint | Done |
| `test.py` | Missing test entry | Added wrapper around validation CLI | Provide explicit testing entry | Done |
| `benchmark.py` | Imported `mekd.validate` and recorded stale `python -m mekd.benchmark` metadata | Switched to local `validate` import and direct-script metadata | Match repository layout and keep help lightweight | Done |
| `export_student.py` | Imported `mekd.*` and lacked checkpoint existence check | Switched to local imports and explicit check | Improve portability and errors | Done |
| `data/build.py` | No eval-only loader or worker seeding | Added `build_dataset`, `build_eval_loader`, worker seeds, generator | Improve evaluation usability and reproducibility | Done |
| `data/datasets.py` | RGB labels without palette were silently converted to grayscale | Raise clear error unless RGB is grayscale-equivalent | Prevent corrupted class IDs | Done |
| `losses.py` | Package-relative import failed for direct scripts | Changed to top-level import | Fix runtime import path | Done |
| `models/mekd.py` | Parent-relative imports failed for direct scripts | Changed to top-level imports | Fix runtime import path | Done |
| `metrics.py` | Precision/Recall missing | Added per-class and mean Precision/Recall | Complete requested metrics | Done |
| `config.py` | Seed controls incomplete | Added deterministic/drop_last/pin_memory defaults | Expose reproducibility knobs | Done |
| `utils.py` | No shared seed helper | Added RNG and DataLoader seed utilities | Improve repeatability | Done |
| `requirements.txt` | Overly strict torch pin and missing NumPy | Relaxed torch requirement, added NumPy | Match code dependencies | Done |
| `configs/default.yaml` | Missing default example config | Added full default YAML | Improve configurability | Done |
| `configs/example_dataset.yaml` | Missing custom dataset example | Added example with RGB palette | Help users adapt data | Done |
| `configs/udd6_mekd.yaml` | RGB palette not configured | Added UDD6 palette | Support RGB labels | Done |
| `README.md` | Referenced non-existent `mekd/` paths and lacked full audit usage | Rewritten | Make repository usable by other researchers | Done |
| `tools/smoke_test.py` | Missing lightweight model check | Added random-tensor smoke test | Check model, loss, backward, metrics without data | Done |
| `tools/check_dataset_structure.py` | Missing dataset checker | Added read-only structure checker | Detect missing labels, bad values, size mismatch, split overlap | Done |
| `mekd.pdf` audit docs | Earlier PDF contained unrelated C2M-Net/autonomous-driving text | Rechecked revised PDF and updated README/audit reports | Remove obsolete paper-file blocker and record remaining scope gaps | Done |
