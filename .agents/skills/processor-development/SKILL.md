---
name: processor-development
description: Create or modify reusable SDM processors and focused tests. Use when adding a processor or changing processor behavior.
---

# Develop an SDM Processor

## Workflow

1. Read the repository-root `AGENTS.md` first and follow any more-specific instructions and matching skills.

2. Inspect the target implementation, neighboring processors, and their tests. Identify the closest existing analogue and follow its naming, lifecycle, state, export, and test patterns. When modifying a processor, first establish its current observable behavior.

3. Confirm the change belongs in the shared processing layer and is reusable across models and recipes. Keep model- or recipe-specific policy outside `sdm.processing`.

4. Treat current SDM processors as the architectural authority. When relevant, consult scikit-learn for established preprocessing API and semantics and cuML for comparable GPU-oriented implementations. Do not add either as a dependency solely for reference.

5. Prefer a regular `Processor` when the operation itself does not require ensemble awareness, relying on `EnsembleProcessorAdapter` for ensemble execution. Implement `EnsembleProcessor` directly only when behavior inherently depends on ensemble members or ensemble structure.

6. Preserve leading batch dimensions according to the existing processor contract.

7. Keep `_fit`, `_transform`, `_fit_transform`, and `_inverse_transform` simple and sequential. Prefer keeping logic in the core method over extracting one-use helpers. Extract helpers only when they express a meaningful concept or provide real reuse.

8. Add focused tests

## Verification

Run the focused processor tests and the repository-required checks for the changed files.
