# Overview

This repository is an open-source model zoo for foundation models on structured data (e.g., TabICLv2, KumoRFM-2, etc).

The repository provides reusable model architectures, tensor containers, preprocessing and postprocessing blocks, attention modules, key/value cache building blocks, ensembling utilities, benchmark examples, and NIM-compatible runtime foundations.
It should stay generic, modular, and lightweight.
Do not add platform or serving abstractions unless explicitly requested.

# Commands

- Test execution via `pytest`
- Pre-commit checks via `pre-commit run --all-files`

# Project Structure

- `schemafm/stype.py`: Semantic column types via `Stype`.
- `schemafm/tensor`: Custom PyTorch-native `Tensor` subclasses for tensorized raw table data.

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
- Add short tensor shape comments for complex tensor operations.
- Avoid accidental graph breaks where a `torch.compile`-friendly formulation is straightforward.
- Follow established naming conventions over ad-hoc names (e.g. prefer the ecosystem-standard `fill_value` to a coined `empty_value`).
- Document public constructor parameters with an `Args:` section in the class docstring, not just a one-line summary.
