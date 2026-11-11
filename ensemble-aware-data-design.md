# Unified ensemble processing

This replaces the original plan in PR #797. It builds on the existing `EnsembleData`, `EnsembleTensor`, and `EnsembleTable`, the registered-buffer support in #1344, and the default tensor type parameter in #1361.

## Contract

- A logical member retains its identity across fit, query, splitting, sharing, and regrouping. Physical groups are an execution optimization, not the identity of fitted state.
- `EnsembleData.execute` owns argument discovery, member alignment, partitioning, physical execution, and result reconstruction. There is no separate planner, selection, storage, or state hierarchy.
- Pure operations may reuse shared values. Randomness, mutation, aliasing, and non-tensor outputs require explicit semantics; they must not silently inherit rules intended for deterministic numerical operations.
- Tensor operations address member-local dimensions. The physical execution batch is hidden, including in reductions, indexing, reshaping, and broadcasting.
- Python values such as column names and ragged vocabularies remain valid ensemble values. They are not padded or converted to tensors merely to make batching convenient.
- Regrouping is permitted when compatible; fitted per-member state must remain correctly aligned. Recipe execution must no longer enforce the old prohibition on merging previously separated groups.

## Execution and tensor interface

Implement execution directly on `EnsembleData.execute`, with fast paths for one input and already aligned inputs. Preserve existing groups or views whenever possible. Refine partitions only when different operands require it; materialize indexing only when the physical operation needs it. Support positional/keyword arguments, nested inputs, mixed plain/ensemble values, and structured outputs.

Extend the existing `EnsembleTensor` rather than introducing a replacement. The plain container uses `__torch_function__`, method delegation and Python operators; `__torch_dispatch__` alone would require a Tensor subclass. Both `torch.mean(value, dim=-2)` and `value.mean(dim=-2)` work at runtime. Runtime protocol support and static typing are distinct: PyTorch's global function stubs do not accept the plain container automatically.

Expose numerical and other semantic blocks on `EnsembleTable` as ensemble-aware values. Replacement constructs tables from the replacement and retained blocks without aligning discarded values. Lift the table operations needed by existing processors through execution, without requiring a generalized `TableTensor` dispatch protocol in this change.

## Processor and recipe migration

Move the existing enhanced buffer implementation to the common Processor lifecycle. Keep `register_buffer`, ordinary attribute assignment, device transfers, persistence, and serialization. Do not introduce a new `State`, `Parameter`, or `Buffer` abstraction.

Use a single fit/transform implementation on plain or ensemble table values. Stochastic tensor callbacks request independent draws explicitly, retaining the original generator draw order. `EnsembleProcessor`, `EnsembleProcessorAdapter` and their ensemble lifecycle methods are deleted. Physical storage iteration and reconstruction belong in containers, not processors. Existing numerical routines that preserve leading batch dimensions run through `execute(..., batched=True)`; native elementwise operations use Torch tags and other logical-axis operations use `vmap`.

Integrate through the real Kumo Tabular recipe and `RecipeExecution`; preserve output dimensions, target inversion, category alignment, missing-value semantics, generators, and model preprocessing parity. General model inference is validation only, not part of the timed benchmark scope.

## Measurement gate

Before changing implementation code, retain raw baseline results at the dependency-integrated commit. Use the unmodified Kumo default recipe, classification targets, 16 estimators as configured by the Kumo benchmark adapter, and additional sizes 1, 4, and 8 on 10,000 × 500. The direct model API's implicit ensemble size for a 2D input is separately documented as one, not confused with the benchmark configuration.

Measure all requested datasets with exactly 89% numerical, 10% categorical, and 1% text columns. Persist generated inputs and their hashes so both versions consume identical values and dictionaries. Text is passthrough in the actual default Kumo recipe; no text embedding is added just for this benchmark.

Report separately:

- Uninstrumented end-to-end recipe fit-transform and query-transform median/p95 latency, unfused feature fit, sampled peak process RSS, and peak allocated GPU memory.
- Per-processor fit, transform, and fit-transform callbacks actually invoked by the recipe in separate instrumented runs, including inclusive/exclusive host time, CUDA-event time, and allocation/copy/indexing counters. Do not sum nested inclusive times or label input sharing as proven redundant arithmetic. Old table-body counts and new ensemble-body counts are different boundaries; compare ATen dispatches and selective kernel traces instead.
- Selective PyTorch Profiler traces for 1,000 × 100 and 10,000 × 500 at 16 estimators, including allocations outside the dense-operation counter's coverage.
- Saved preprocessing outputs for per-member, schema, category, text, numerical, and target parity checks after migration.

Use the same hardware, environment, thread count, warmups, repetitions, task, fixture hashes, and timing boundaries before and after. Keep raw measurements, including failures and unavailable metrics. Do not infer savings from code shape alone.

## Delivery gates

1. Preserve complete baseline matrix and representative profiles before implementation changes.
2. Establish execution, arbitrary-value, tensor-operation, dimension, aliasing, and reconstruction regression tests.
3. Migrate every processor and the recipe lifecycle; remove obsolete APIs and restrictions.
4. Pass focused and shared contracts, device/state/serialization tests, and end-to-end Kumo preprocessing parity.
5. Repeat the full benchmark matrix and optimize demonstrated bottlenecks. Report regressions as well as speedups; do not introduce padding or dynamic rebatching without measured justification.

## Current local status

The unified implementation, processor migration, dependency integration, complete nine-case correctness matrix, state/device/serialization contracts, actual Kumo classification/regression prediction parity and fullgraph Kumo prediction tests are implemented. Fused overrides remain only for category-fit computation reuse and composite routing that must preserve child fusion. The original baseline is immutable, and reproducible code plus raw per-processor allocation/operation comparisons are available.

The full matched nine-case before/after matrix is complete on an otherwise idle L4. All primary-run medians improve; a separate 21-repeat check still shows a 2.2% E1 median regression and a 1.2% E4 improvement. E16 peak GPU memory increases about 0.9–1.7%, so the lower-memory goal remains unmet. Genuine selective CUDA profiles, CPU/CUDA prediction parity and per-processor timing/allocation/peak-memory reports are available. Heterogeneous schemas retain unsupported members unchanged, including inverse transforms. The benchmark refuses latency measurements while another process occupies the GPU. See [benchmark/ensemble/PERFORMANCE.md](benchmark/ensemble/PERFORMANCE.md) for measurements, limitations and the publication gate. This replacement plan is local and has not been published to GitHub without user review.
