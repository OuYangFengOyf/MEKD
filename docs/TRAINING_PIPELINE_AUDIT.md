# Training Pipeline Audit

## Basic Logic

The training loop performs:

1. `model.train()`.
2. Expert/teacher eval mode during distillation.
3. `optimizer.zero_grad(set_to_none=True)`.
4. Forward pass.
5. Loss computation.
6. AMP-scaled backward.
7. Optimizer step.
8. Scheduler step.
9. Validation.
10. Save `last.pth` and `best.pth`.

## Optimizer and Scheduler

- Optimizer: AdamW.
- LR: configurable, default `4e-4`.
- Weight decay: configurable, default `1e-2`.
- Scheduler: warm-up and cosine LambdaLR.
- Stage transition: optimizer is rebuilt after expert freezing.

## Device Handling

- `cfg.device` is used when available.
- If CUDA is requested but unavailable, the code falls back to CPU.
- No hard-coded `.cuda()` calls were found.

## Random Seed

Fixed during audit:

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- DataLoader workers
- DataLoader generator

`train.deterministic` can set cuDNN deterministic mode.

## Save Logic

- `train.save_dir` is created automatically.
- `config.resolved.json` is saved.
- `last.pth` and `best.pth` are saved.
- Resume checkpoint existence is checked before loading.

## Remaining Notes

- Scheduler state is reconstructed from epoch index rather than saved explicitly. This is acceptable for epoch-boundary resume but not mid-epoch resume.
- Current code does not implement the optional self-distillation fine-tuning stage described in the paper.
