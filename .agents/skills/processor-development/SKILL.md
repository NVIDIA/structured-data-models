---
name: processor-development
description: Create or modify reusable SDM processors and focused tests. Use when adding a processor, changing fit/transform/inverse behavior, or deciding how a processor participates in ensembles; not for preprocessing specific to one model recipe.
---

# Develop an SDM Processor

## Workflow

1. Read the repository-root `AGENTS.md` before inspecting or editing processor code, then follow any more-specific instructions and matching skills.
2. Inspect the target implementation, neighboring processors, and their tests. Identify the closest analogue and follow its naming, lifecycle, state, export, and test patterns. When modifying a processor, first establish its current observable behavior.
3. Confirm the change is reusable across models and recipes. Keep one-off or model-specific policy in the relevant model or recipe instead of `sdm.processing`.
4. Use current SDM processors as the architectural authority. Use scikit-learn as the reference for established API names and fit/transform semantics, and cuML as a GPU-oriented reference when a comparable implementation is relevant. Do not add either as a dependency solely for reference.
5. Prefer a regular `Processor` and let `EnsembleProcessorAdapter` provide per-ensemble-group behavior. Implement `EnsembleProcessor` directly only when behavior truly depends on estimator membership or changes ensemble structure.
6. Preserve every leading batch dimension and process batches independently. Keep tensor work on-device and preserve dtype unless the public behavior requires otherwise.
7. Keep `_fit`, `_transform`, `_fit_transform`, and `_inverse_transform` as simple sequential code. Do not extract one-use helper functions unless they name a meaningful invariant or enable real reuse.
8. Add focused tests beside the closest analogue for every changed behavior. Include leading-batch coverage and ensemble coverage when relevant, and assert public results rather than private structure.

## Verification

Run the focused processor tests and pre-commit checks for the changed files.
