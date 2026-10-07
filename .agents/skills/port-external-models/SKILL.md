---
name: port-external-models
description: Implement or review a port of an external model in SDM, from upstream inference behavior and checkpoints through public prediction parity and reviewable PRs.
---

# Port an External Model to SDM

Follow the repository-root `AGENTS.md`. When a port adds reusable processors or changes a `Recipe`, also follow the `processor-development` or `recipe-development` skill, respectively. Use `docstring` when writing or reviewing public docstrings.

## Establish the reference

- Which revision: Pin the upstream git SHA, Hub revision, model config, and source vs weight licenses. Put them in the PR description; when merging, also in `THIRD_PARTY_LICENSES.md`, the model docstring, and the checkpoint load path.
- Which behaviour in this revision: Trace an upstream **public inference call** through input preparation, numerical execution, and output restoration. Identify the functions responsible for each stage; a low-level `forward` method alone does not establish public behavior.
- Which public contract: Record accepted forms and shapes, feature and target roles, auxiliary inputs, batching and size limits, missing and non-finite values, masks or padding, preprocessing and output restoration, and output ordering where applicable. Include any task-specific semantics that affect predictions. Distinguish public boundary behavior from numerical model computation, and state which capabilities the SDM port supports.

## Build the SDM path

- Before porting blocks, specify the SDM `forward`: which public inputs become which core tensors, and how core outputs become the returned `TableTensor`.
- Keep the model wrapper responsible for SDM inputs, outputs, and lifecycle. Put reusable numerical operations in `sdm.nn`, model-specific composition and checkpoint handling in model components, and independently reusable preprocessing in `sdm.processing`.
- `ICLModel._forward` takes `TableTensor` (plus related tables when used). Unwrap to `Tensor` for numerical work and keep that tensor's device and dtype. `sdm.nn` takes `Tensor` only; keep `TableTensor` only while column or relation schema is still required.
- Start from existing `sdm.nn` components. Adapt one to the required semantics when the change is reusable across models; otherwise add a reusable component for a distinct operation. Prefer an efficient, numerically equivalent formulation over copying upstream computations, and verify it against the reference.
- Design components for clear SDM semantics, not for strict loading of an upstream state dict. Remap checkpoint keys or tensors at the loading boundary as needed, and verify that all weights required by supported inference are loaded.
- Write computations in execution order with short, typed methods and meaningful intermediate names. Extract helpers for distinct behavior or real reuse; omit upstream factories, configuration layers, and wrappers that SDM does not need.
- Compose default recipes from existing public processors. Use a fitted recipe transform only when its state can be learned from permitted data and applied without leakage; ensure any inverse transform works on the prediction's actual shape.

## Plan reviewable PRs

Split the port into this stack. These are phases, not five PRs: Blocks and Capabilities are one PR each. Title PRs `[Model N/n]` with `n` the total count. Each PR has one reviewable outcome, names its parent PR, and keeps the branch working. Treat roughly 300 changed lines as a prompt to split, not a reason for incomplete PRs.

1. **Skeleton (optional):** package and `ICLModel` subclass, imported only from the submodule. Zeros are allowed here.
2. **Blocks:** one numerical piece per PR (`sdm.nn` or model component) with a focused tensor test. Add only what the working path needs.
3. **Working path:** assemble the core, load the pinned checkpoint, and run one public `forward`/`predict` for one supported input kind (related tables only if required). The prediction matches the pinned reference; a public behavior test covers it. Zeros, stubs, or block-only tests do not complete this phase.
4. **Capabilities:** extra input types, auxiliaries, missing values, size limits, recipe details. Keep each capability's implementation, tests, and needed docs together.
5. **Index (last):** add the model to `sdm.models` and the docs index.

## Verify and review

- Generate deterministic reference outputs from the pinned upstream revision. Compare observable SDM predictions through the public path, including checkpoint loading and output ordering. Claim end-to-end parity only after this comparison.
- Test supported boundaries and edge cases, such as missing or constant inputs, multiple outputs, auxiliary data, and size limits where applicable. Assert public behavior rather than internal helper layout.
- Benchmark performance-sensitive GPU changes with synchronization-aware timing.
- When reviewing a proposed port, reconstruct the upstream public contract and intended SDM call path before evaluating the diff. Check the working public slice, reference evidence, leakage boundaries, checkpoint behavior, and parity claims against that contract; suggest removing machinery unrelated to the supported path.
