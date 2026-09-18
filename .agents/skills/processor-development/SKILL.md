---
name: processor-development
description: Create or modify reusable SDM processors and focused tests. Use when adding a processor or changing processor behavior.
---

# Develop an SDM Processor

## Workflow

01. **General**: Read the repository-root `AGENTS.md` first and follow any more-specific instructions and matching skills.

02. **Keep processors general**: Before adding behavior to a processor, check whether it describes the processing operation itself or a specific model/recipe use case. Keep only the former in sdm.processing.

03. **Learn from existing processors**: Before implementing, inspect the target processor and a few relevant existing processors and tests. Reuse established SDM API & coding patterns rather than introducing a new approach. When modifying an existing processor, understand its current behavior and tests before changing it. For standard preprocessing operations, check the corresponding scikit-learn API and behavior before designing a new interface. Consult cuML when a comparable GPU implementation exists and its implementation is useful for SDM.

04. **Choose the simplest processor type**: Use a regular `Processor` when its state and behavior are defined per table or batch; `EnsembleProcessorAdapter` handles its use with ensembles. Implement `EnsembleProcessor` directly only when state or behavior is scoped to logical ensemble members, depends on multiple members, or changes the ensemble structure.

05. **Keep state and the public API minimal**: Keep fitted state private and PyTorch-native, using registered buffers and existing state containers. Do not expose processor state through new properties or methods unless required by existing public behavior. Do not treat implementation details as public API merely because existing tests access them.

06. **Leading dimensions**: Regular processors must support arbitrary leading batch dimensions without changing their semantics.

07. **Keep the main flow local**: Keep processor-specific logic in `_fit`, `_transform`, `_fit_transform`, and `_inverse_transform` simple and sequential. Do not add defensive validation, compatibility helpers, or convenience abstractions unless they are required by the processor's specific public behavior. Rely on AGENTS.md and the shared processor/ensemble abstractions for generic contracts and validation. Use the lifecycle behavior provided by the base classes. Override methods such as `_fit_transform` only when the processor needs different behavior or can reuse substantial computation. Avoid moving small pieces into helpers or compressing branching and multi-step logic into expressions merely to shorten the method.

08. **Backward compatibility**: Processor APIs and internal state do not need to remain backward compatible. Do not preserve old attributes, properties, parameters, or behavior solely for compatibility. When the design changes, update affected tests and callers instead of adding compatibility layers.

09. **Defaults require justification**: Never choose a default because it seems reasonable. Derive defaults from established semantics or evidence: first from the existing SDM contract or closest analogue, then from the established external API/reference when applicable. Prefer a neutral/no-op default when that preserves existing behavior. If no defensible default exists, require the argument instead of inventing one.

10. **Docs:** Keep documentation minimal and proportional to the change. Document the public behavior and parameters, but do not add explanatory sections or implementation details unless they are necessary to understand how to use the processor. Do not explain ensemble grouping, member sharing, or how the operation is applied across ensemble members; that belongs on `EnsembleProcessor` and `EnsembleTable`. Mention ensemble members only when the public operation itself is routing or reducing them.

11. **Tests:** Test shared Processor behavior through the processor contract tests and register new processors there as applicable. Keep processor-specific tests limited to behavior unique to that processor. No tests on validation.

## Verification

Run the focused processor tests and the repository-required checks for the changed files.

## Review

When reviewing an existing processor change, do not take the proposed implementation or its documentation as the design baseline.

1. Reconstruct the intended behavior from the PR/task and existing public contracts.
2. Determine how you would implement and document that behavior from scratch following this skill.
3. Compare that minimal design with the proposed diff.
4. Flag code that exists only because of the proposed implementation rather than because the behavior requires it, including unnecessary state, validation, compatibility layers, helpers, or public API.
5. Write the docstring from scratch and compare it with the proposed text instead of editing the proposed text. Flag sentences that restate the summary, repeat contracts already documented on `Processor`/`EnsembleProcessor` such as batch or member independence, describe raised errors or missing validation, or state that untouched blocks stay unchanged.
6. Check whether existing wording or tests caused implementation details to be preserved. Neither tests nor docstrings copied from sibling processors make internal behavior part of the public contract.
7. Suggest removing unnecessary code and documentation, not only modifying it.
