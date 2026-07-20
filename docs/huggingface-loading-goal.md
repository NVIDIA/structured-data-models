# Hugging Face Loading Goal

Support pretrained model weights hosted on Hugging Face through one shared,
model-agnostic loading path.

## Current State

- `TabICLv2` downloads fixed checkpoint files from a fixed Hugging Face repo.
- `KumoRFM` exposes a `pretrained` argument but does not yet load weights.
- Model wrappers own checkpoint remapping and `load_state_dict` behavior.

## Goal

Add a small helper that resolves checkpoint files from Hugging Face while
keeping model-specific architecture and state-dict handling inside each model
wrapper.

Expected model constructors should be able to expose:

- `pretrained`: whether to load default weights.
- `repo_id`: optional Hugging Face repository override.
- `revision`: optional branch, tag, or commit SHA for reproducible loading.
- `cache_dir`: optional cache location.
- `local_files_only`: whether to avoid network access.

## Design Principles

- Keep the public API Python-first and lightweight.
- Preserve existing default behavior for current pretrained models.
- Prefer pinned revisions in examples and tests when exact weights matter.
- Let `huggingface_hub` handle authentication, tokens, cache layout, and
  offline behavior.
- Avoid adding model registry or serving abstractions.

## First Implementation Step

Introduce the shared file resolver, move `TabICLv2` onto it, and add KumoRFM
checkpoint loading by providing default repo metadata and filenames from the
model wrapper.
