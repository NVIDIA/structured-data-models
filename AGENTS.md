# Overview

This repository is an open-source model zoo for structured data models (e.g., TabICLv2, KumoRFM-2, etc).

The repository provides reusable model architectures, tensor containers, preprocessing and postprocessing blocks, attention modules, key/value cache building blocks, ensembling utilities, benchmark examples, and NIM-compatible runtime foundations.
It should stay generic, modular, and lightweight.
Do not add platform or serving abstractions unless explicitly requested.

# Commands

- Test execution via `pytest`
- Pre-commit checks via `pre-commit run --all-files`

# PR / GitHub Metadata

- Do not mention Codex, AI, or tool attribution in PR titles, PR descriptions, commit messages, or review replies unless explicitly requested.
- PR metadata should describe the code change only.

# Project Structure

- `sdm/stype.py`: Semantic column types via `Stype`.
- `sdm/cache.py`: Model cache, e.g., for key/value caching.
- `sdm/tensor`: Custom PyTorch-native `Tensor` subclasses for tensorized raw table data.
- `sdm/processing`: Common tensorized preprocessing and postprocessing routines for structured data models.
- `sdm/nn`: Common neural network building blocks for structured data models.

# Core Design Principles

- Keep the project PyTorch/tensor-centric.
- Preserve dataframe ergonomics at the boundary, but move model execution onto structured tensor containers.
- Keep model-family wrappers thin. Shared abstractions should live outside model implementations if possible.
- Avoid mandatory config-first APIs. Direct Python composition should be the primary interface.
- Add composable transformations instead of hard-coding one-off preprocessing into model wrappers.
- Keep recipes inspectable and deterministic where possible. Any stochastic transformations should expose seed/generator control.
- Treat preprocessing as leakage-sensitive.
  Transformations that learn state must be scoped to the context/training portion unless explicitly designed otherwise.
- Keep dependencies minimal in the core package.
  Heavy dependencies should be optional unless they become essential.
- Aim for GPU acceleration in all core components. Prefer PyTorch and cuDF execution paths over CPU-bound pandas, NumPy, or sklearn implementations.

# Python/PyTorch Coding Style

- Keep Python code typed at function and method boundaries.
- Use keyword arguments in multi-line calls.
- Avoid `else` after `return`, `raise`, `break`, or `continue`.
- Prefer PyTorch-native, vectorized tensor operations over NumPy or Python loops.
  Call out cases where vectorization is not practical.
- Preserve tensor device and dtype.
  Avoid accidental transfers through `.cpu()`, `.numpy()`, `.item()`, Python scalars, or newly-created CPU tensors.
- Prefer idiomatic PyTorch broadcasting and tensor methods.
  Avoid materializing same-shaped helper tensors or explicit views when a scalar or lower-rank tensor broadcasts clearly, e.g. prefer `torch.where(mean.isnan(), fill_value, mean)` over creating a `mean.new_full(...)` placeholder first.
- Add short tensor shape comments for complex tensor operations.
- Avoid accidental graph breaks where a `torch.compile`-friendly formulation is straightforward.
- Use established names.
- Document public constructor parameters.
