# Ensemble-Aware Processing: Required Implementation Extensions

The pre-implementation [design](ensemble_aware_processing_design.md) and [problem statement](ensemble_aware_processing_problem.md) remain the normative specification and are not changed to match the implementation. The following narrow extensions are required by reference parity or the public model contract.

## TabICLv2 Paired Member Plan

- The generic contract derives stochastic decisions from member, table-scope, and Processor-path streams.
- TabICLv2 additionally pairs normalization, feature permutations, and class permutations in one reference ensemble plan. Independent Processor streams cannot reproduce those cross-path pairings.
- A private TabICLv2 plan therefore supplies the paired permutations through EnsembleFitContext. It is not public Recipe metadata, and generic Choice and shuffle semantics remain unchanged.

## TabICLv2 Positional Model Materialization

- Generic EnsembleTable materialization requires identical schema and categorical metadata.
- TabICLv2 consumes feature positions and categorical codes, while TargetDecode retains the semantic output mapping. Member-specific shuffles intentionally change names or category order without changing the model-core layout.
- TabICLv2 therefore owns a private positional grouping and materialization override based on shape, stype layout, dtype, device, and category counts. Other models retain strict generic materialization.

## Model Execution Scheduling

- Recipe execution produces one compact provenance-aware ensemble plan regardless of model schedule.
- ICLModel exposes parallel, sequential, and automatic model execution. Sequential execution materializes one model member at a time; automatic `forward` and `fit` execution retry model-core CUDA out-of-memory failures sequentially. Cached `predict` reuses the schedule established during `fit`.
- Recipe preprocessing itself remains shared and materializes all required variants. Streaming one member through the entire Recipe would require a different fitted-state and output contract and is not part of the version-1 design.

## Variable-Schema Inputs With Existing Leading Dimensions

- Version 1 accepts a logical Recipe table with shape [R, C] when a VariableSchemaBatchMixin is present. The adapter may split the ensemble variant dimension because EnsembleTable maps that dimension back to members.
- A logical input shaped [..., R, C] could produce a different schema at each existing leading position. One TableTensor cannot represent those different schemas, and EnsembleTable has no mapping for non-member positions. Such inputs are rejected until that representation is designed explicitly.
