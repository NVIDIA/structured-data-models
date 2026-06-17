# Overview

This repository is an open-source model zoo for foundation models on structured data (e.g., TabICLv2, KumoRFM-2, etc).

The repository provides reusable model architectures, tensor containers, preprocessing and postprocessing blocks, attention modules, key/value cache building blocks, ensembling utilities, benchmark examples, and NIM-compatible runtime foundations.
It should stay generic, modular, and lightweight.

# Commands

- Test execution via `pytest`
- Pre-commit checks via `pre-commit run --all-files`

# Core Design Principles

- Keep the project PyTorch/tensor-centric.
- Preserve dataframe ergonomics at the boundary, but move model execution onto
  structured tensor containers.
- Keep model-family wrappers thin. Shared abstractions should live outside
  TabICL, KumoRFM, or any one model implementation.
- Make context/query boundaries explicit in public structured objects, even
  when model internals require contiguous packed rows.
- Avoid mandatory config-first APIs. Direct Python composition should be the
  primary interface.
- Add composable transforms instead of hard-coding one-off preprocessing into
  model wrappers.
- Keep recipes inspectable and deterministic where possible. Any stochastic
  transform should expose seed/generator control.
- Treat preprocessing as leakage-sensitive. Transforms that learn state must be
  scoped to the context/training portion unless explicitly designed otherwise.
- Keep dependencies minimal in the core package. Heavy stacks such as cuDF,
  PyG, `pyg-lib`, explainability libraries, or benchmark tooling should be
  optional extras unless they become essential.
- Prefer GPU acceleration where it matters, but keep CPU/pandas interop usable.
- Do not introduce serving/product abstractions unless specifically requested.

# Model Integration Notes

TabICLv2:

- Public reference implementations use sklearn-style `fit(X_train, y_train)`
  and `predict(X_test)`.
- Raw model execution expects packed rows: context/training rows first, followed
  by query/test rows.
- `y_train` determines the context length internally.
- KV-cache support separates cached context computation from repeated query
  inference.

TabPFN:

- Public reference implementations also expose sklearn-style `fit`/`predict`.
- Inference code stores train inputs/targets, preprocesses test inputs, then
  concatenates train and test rows before model execution.
- Cache paths may execute with test-only inputs plus cached context state.

For `schema-fm`, preserve the useful parts of these designs while exposing a
lower-level model-zoo interface:

- public structured objects should know target, context rows, and query rows;
- model adapters may pack or concatenate internally;
- caching should be a shared concept where possible, not duplicated per model
  family.

# References

- [TabICLv2](https://arxiv.org/abs/2602.11139)
- [TabICL repository](https://github.com/soda-inria/tabicl)
- [TabPFNv2](https://www.nature.com/articles/s41586-024-08328-6)
- [TabPFN repository](https://github.com/PriorLabs/TabPFN)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [transformers](https://github.com/huggingface/transformers)
- [sentence-transformers](https://github.com/huggingface/sentence-transformers)
