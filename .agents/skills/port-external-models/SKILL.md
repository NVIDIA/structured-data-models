---
name: port-external-models
description: Implement or review a port of an external model in SDM, from upstream inference behavior and checkpoints through public prediction parity and reviewable PRs.
---

# Port an External Model to SDM

Follow the repository-root `AGENTS.md`. When a port adds reusable processors or changes a `Recipe`, also follow the `processor-development` or `recipe-development` skill, respectively. Use `docstring` when writing or reviewing public docstrings.

## Establish the reference

- Pin the upstream source commit, checkpoint revision, model configuration, and applicable source and weight licenses. Record these in the port's review material or documentation so the comparison is reproducible.
- Trace an upstream **public inference call** through input preparation, numerical execution, and output restoration. Identify the functions responsible for each stage; a low-level `forward` method alone does not establish public behavior.
- Record the reference's public input and output contract: accepted forms and shapes, feature and target roles, auxiliary inputs, batching and size limits, missing and non-finite values, masks or padding, preprocessing and output restoration, and output ordering where applicable. Include any task-specific semantics that affect predictions. Distinguish public boundary behavior from numerical model computation, and state which capabilities the SDM port supports.

## Build the SDM path

- Design the public call path before porting numerical helpers: specify how each input reaches the model core and how predictions return through the public API.
- Keep the model wrapper responsible for SDM inputs, outputs, and lifecycle. Put reusable numerical operations in `sdm.nn`, model-specific composition and checkpoint handling in model components, and independently reusable preprocessing in `sdm.processing`.
- Accept `TableTensor` at the public boundary, convert once to ordinary PyTorch tensors for numerical operations, and preserve device and dtype through the model.
- Start from existing `sdm.nn` components. Adapt one to the required semantics when the change is reusable across models; otherwise add a reusable component for a distinct operation. Prefer an efficient, numerically equivalent formulation over copying upstream computations, and verify it against the reference. Translate checkpoint keys explicitly instead of shaping the public model structure around upstream parameter names.
- Write computations in execution order with short, typed methods and meaningful intermediate names. Extract helpers for distinct behavior or real reuse; omit upstream factories, configuration layers, and wrappers that SDM does not need.
- Compose default recipes from existing public processors. Use a fitted recipe transform only when its state can be learned from permitted data and applied without leakage; ensure any inverse transform works on the prediction's actual shape.

## Plan reviewable PRs

- Start with the smallest working path through the public API: a supported input runs real model computation and returns a correct prediction, verified by a public behavior test. Placeholder predictions do not complete a feature slice.
- If that path needs substantial machinery, put prerequisite blocks in separate PRs only when each has a clear purpose and focused behavioral test. Add only what the working path needs.
- Extend the working path in coherent capabilities, such as additional input types, auxiliary data, missing-value behavior, or larger supported sizes. Keep each capability's implementation, tests, and necessary documentation together.
- Give each PR one reviewable outcome, name any dependency on an earlier PR, and keep intermediate branches working. Treat roughly 300 changed lines across code, tests, and documentation as a prompt to seek a coherent split, not a target or a reason for incomplete PRs.

## Verify and review

- Generate deterministic reference outputs from the pinned upstream revision. Compare observable SDM predictions through the public path, including checkpoint loading and output ordering. Claim end-to-end parity only after this comparison.
- Test supported boundaries and edge cases, such as missing or constant inputs, multiple outputs, auxiliary data, and size limits where applicable. Assert public behavior rather than internal helper layout.
- Benchmark performance-sensitive GPU changes with synchronization-aware timing.
- When reviewing a proposed port, reconstruct the upstream public contract and intended SDM call path before evaluating the diff. Check the working public slice, reference evidence, leakage boundaries, checkpoint behavior, and parity claims against that contract; suggest removing machinery unrelated to the supported path.
