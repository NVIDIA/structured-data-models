---
name: processor-development
description: Create or modify reusable SDM processors and focused tests. Use when adding a processor or changing processor behavior.
---

# Develop an SDM Processor

## Workflow

1. **General**: Read the repository-root `AGENTS.md` first and follow any more-specific instructions and matching skills.

2. **Keep processors general**: Before adding behavior to a processor, check whether it describes the processing operation itself or a specific model/recipe use case. Keep only the former in sdm.processing.

3. **Learn from existing processors**: Before implementing, inspect the target processor and a few relevant existing processors and tests. Reuse established SDM API & coding patterns rather than introducing a new approach. When modifying an existing processor, understand its current behavior and tests before changing it. For standard preprocessing operations, check the corresponding scikit-learn API and behavior before designing a new interface. Consult cuML when a comparable GPU implementation exists and its implementation is useful for SDM.

4. **Choose the simplest processor type**: Use a regular `Processor` when its state and behavior are defined per table or batch; `EnsembleProcessorAdapter` handles its use with ensembles. Implement `EnsembleProcessor` directly only when state or behavior is scoped to logical ensemble members, depends on multiple members, or changes the ensemble structure.

5. **Keep state and the public API minimal**: Keep fitted state private and PyTorch-native, using registered buffers and existing state containers. Do not expose processor state through new properties or methods unless required by existing public behavior.

6. **Leading dimensions**: Regular processors must support arbitrary leading batch dimensions without changing their semantics.

7. **Keep the main flow local**: Keep processor-specific logic in `_fit`, `_transform`, `_fit_transform`, and `_inverse_transform` simple and sequential. Do not add defensive validation, compatibility helpers, or convenience abstractions unless they are required by the processor's specific public behavior. Rely on AGENTS.md and the shared processor/ensemble abstractions for generic contracts and validation. Use the lifecycle behavior provided by the base classes. Override methods such as `_fit_transform` only when the processor needs different behavior or can reuse substantial computation.

8. **Docs:** Keep documentation minimal and proportional to the change. Document the public behavior and parameters, but do not add explanatory sections or implementation details unless they are necessary to understand how to use the processor.

9. **Tests:** Test shared Processor behavior through the processor contract tests and register new processors there as applicable. Keep processor-specific tests limited to behavior unique to that processor. When a change fixes a bug, add a focused regression test that reproduces the bug and verifies the fix.

## Verification

Run the focused processor tests and the repository-required checks for the changed files.
