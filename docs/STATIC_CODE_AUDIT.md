# Static Code Audit

## Imports

- Original issue: entry scripts imported `mekd.*`, but the repository root is not a `mekd/` package.
- Fix: entry scripts now import local modules such as `config`, `data`, `losses`, `metrics`, and `models`.
- Original issue: `losses.py` used a package-relative import `.priors`, which fails for top-level script execution.
- Fix: changed to `from priors import ...`.
- CLI help no longer imports PyTorch at module import time for `train.py`, `validate.py`, `test.py`, `benchmark.py`, or `export_student.py`.

## Paths

- No `/home/...`, `/media/...`, `C:\...`, `D:\...`, fixed usernames, or author-local dataset roots were found in code/configs.
- Dataset and output paths are configurable through YAML.
- Output directories are created before writing checkpoints or JSON files.

## Dead Code

- No `pdb.set_trace()`, `breakpoint()`, `assert False`, or forced `exit()` was found.
- Minor unused imports were removed from `losses.py`, `models/layers.py`, and `priors.py`.

## Exception Handling

- Empty dataset discovery raises a clear `FileNotFoundError`.
- Missing resume checkpoint and evaluation checkpoint now raise clear `FileNotFoundError`.
- RGB label masks without a palette now raise a clear `ValueError` unless they are grayscale-equivalent RGB masks.

## Remaining Static Risks

- PyTorch is not installed in the current execution environment, so model import/forward checks cannot complete here.
- `__init__.py` at the repository root is not used by the direct script workflow.
