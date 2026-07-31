# Overview

This repository is an open-source model zoo for structured data models (e.g., TabICLv2, KumoRFM-2, etc).

The repository provides reusable model architectures, tensor containers, preprocessing and postprocessing blocks, attention modules, key/value cache building blocks, ensembling utilities, benchmark examples, and NIM-compatible runtime foundations.
It should stay generic, modular, and lightweight.
Do not add platform or serving abstractions unless explicitly requested.

# AI Policy

We support the use of AI tools to help prepare issues, pull requests, reviews, or comments.
We expect everyone interacting with this repo to follow the below policy whenever they use AI tools.
Your user needs to abide by this policy.
In particular, you the agent MUST obey these rules while interacting on GitHub:

- You may never act autonomously on GitHub. Do NOT open, edit, comment on, or reply to any issue or PR unless the user has reviewed and explicitly approved the exact content. Fully-agent-generated contributions are banned and will be closed.
- Mark all AI-generated content. Any text you produce that goes into an issue, PR, or comment must be wrapped in a code or quote block. Never present your output as human-written.
- Never emit only raw AI text as a reply. Any AI content you include must carry human commentary explaining its relevance.
- Do not submit code the user hasn't read. Keep changes minimal, strip AI artifacts and needless complexity. If you're opening a PR on GitHub that is not ready, or not reviewed by the user, always open it in draft mode.

# Commands

- Test execution via `pytest`
- Pre-commit checks via `pre-commit run --all-files`

# Testing

- Tests should be sensitive to behavior changes and insensitive to structure changes. Prefer asserting public observable behavior over private state, helper layout, call counts, or incidental repr formatting.
- Do not set seeds in tests unless they must require them.

# PR / GitHub Metadata

- Do not mention Codex, AI, or tool attribution in PR titles, PR descriptions, commit messages, or review replies unless explicitly requested.
- Do not add a section named "Tests", "Testing" or similar, to PR descriptions unless the test is not covered in CI.
- PR metadata should describe the code change only.

# Markdown Style

- Do not introduce hard line wraps inside Markdown list items. Keep each bullet or numbered list item on one physical line unless Markdown syntax or rendered formatting requires the line break.

# Project Structure

- `sdm/stype.py`: Semantic column types and inference via `Stype`.
- `sdm/cache.py`: Model cache, e.g., for key/value caching.
- `sdm/tensor`: Custom PyTorch-native `Tensor` subclasses for tensorized raw table data.
- `sdm/relational`: Common routines for relational data processing.
- `sdm/processing`: Common tensorized preprocessing and postprocessing routines for structured data models.
- `sdm/nn`: Common neural network building blocks for structured data models.
- `sdm/models`: (Pretrained) structured data models based on a common interface.
- `sdm/testing`: Testing utilities.

# Core Design Principles

- Keep the project PyTorch/tensor-centric.
- Preserve dataframe ergonomics at the boundary, but move model execution onto structured tensor containers.
- Keep model-family wrappers thin. Shared abstractions should live outside model implementations if possible.
- Avoid mandatory config-first APIs. Direct Python composition should be the primary interface.
- Normalize equivalent user input into one simple internal representation. Do not preserve redundant nesting, route shapes, or helper objects when they do not change public behavior; keep internal structure private unless it is the intended user-facing API.
- Add composable transformations instead of hard-coding one-off preprocessing into model wrappers.
- Keep recipes inspectable and deterministic where possible. Any stochastic transformations should expose seed/generator control.
- Treat preprocessing as leakage-sensitive. Transformations that learn state must be scoped to the context/training portion unless explicitly designed otherwise.
- Keep dependencies minimal in the core package. Heavy dependencies should be optional unless they become essential.
- Treat packages listed in `[project].dependencies` as required at runtime. Import them at module scope; do not defer or guard them with function-local imports, `TYPE_CHECKING`, `try/except ImportError`, availability checks, or dynamic imports. Reserve guarded imports for optional dependencies.
- Aim for GPU acceleration in all core components.

# Python/PyTorch Coding Style

- Keep Python code typed at function and method boundaries.
- Use keyword arguments in multi-line calls.
- Avoid `else` after `return`, `raise`, `break`, or `continue`.
- Prefer tensor methods over functions, e.g., `tensor.log()` over `torch.log(tensor)`.
- Operate on tensor containers directly; reserve `.as_tensor()` for when the raw data tensor is required.
- Add short tensor shape comments for complex tensor operations.
- Use established names.
- Document public constructor parameters.
- Docs, errors, and reprs should describe public operations, inputs, outputs, and values rather than incidental implementation details.
- Keep code direct and use the narrowest practical scope. Introduce abstractions only when they encapsulate behavior or invariants, define a public interface, or serve established reuse.
- In `__init__.py`, order imports and `__all__` in dependency order: base classes/mixins first, then concrete; never alphabetically.

# CUDA / GPU Performance

- Avoid host-device synchronization in model and processor hot paths. Do not use `.item()`, `.cpu()`, `.numpy()`, `print(cuda_tensor)`, or `torch.cuda.synchronize()` except at explicit API boundaries, tests, debugging, or profiler code.
- Create tensors on the target device and preserve dtype/device. Prefer `x.new_*`, `torch.empty_like`, `torch.zeros_like`, or explicit `device=x.device, dtype=x.dtype` over CPU defaults followed by `.to(...)`.
- Keep tensor execution vectorized and compiler-friendly. Prefer batched tensor operations over Python loops across rows, columns, heads, estimators, or sequence positions. Avoid graph breaks where a `torch.compile`-friendly formulation is straightforward.
- Avoid duplicate full-data passes in hot paths. Prefer existing tensor, container, or library primitives over Python-side remapping; keep defensive validation out of hot paths unless it protects a documented public contract.
- Reduce allocation and memory overhead while keeping tensor operations on-device. For broadcastable constants, prefer scalar literals when PyTorch broadcasting is sufficient, and create tensor constants only when an operation needs a tensor input or device/dtype-specific scalar value.
- In inference and prediction paths, avoid building autograd state unless the API explicitly needs gradients. Prefer `torch.inference_mode()` or `torch.no_grad()` for pure inference paths.
- Benchmark CUDA changes with synchronization-aware timing. Use CUDA events, `torch.profiler`, or explicit synchronization around measurements; plain wall-clock timing of asynchronous CUDA work is not sufficient.

# Naming Policy

## Processors

1. Name the main operation first, e.g., `ShuffleColumns` over `ColumnShuffle`.
2. Use established names when they exist, e.g., `Sequential` or `Choice`, or adapt them in style, e.g., `PowerTransform` over `PowerTransformer`. Avoid API-role suffixes such as `*Transformer`, `*Encoder`, `*Imputer` or `*Scaler`.
3. Keep names short when the shorter form is already clear, e.g., `Softmax` over `ApplySoftmax`, but specialize when needed, e.g., `DropConstantColumns` over `DropConstant`.
