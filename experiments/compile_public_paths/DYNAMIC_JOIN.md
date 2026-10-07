# Variable-output join compilation prototype

The identifier-key path packages ordinary, nullable, and string column storage as tensor leaves and invokes the existing Arrow/cuDF join through a custom operator. The fake implementation gives both returned index tensors the same dynamic length. Arrow/cuDF remains external code; the compiler does not optimize its join algorithm. Other semantic column types retain the existing eager boundary.

Use a scoped `torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)` around compilation and execution. This option is necessary because duplicate keys, missing keys, and empty inputs change the number of result pairs without changing the input shapes.

Run `PYTHONPATH=. python experiments/compile_public_paths/dynamic_join.py`. CPU Inductor with `fullgraph=True, dynamic=True` passes 24 comparisons on each of PyTorch 2.7.1 and 2.14: duplicate keys, empty left/right, no matches, NaN and numeric casting, nullable IDs, string/null IDs, and multi-column keys, each with int64/int32/uint8 indices. Pair order and dtype match the existing join exactly in those cases. Existing join tests pass (6 passed, 5 CUDA skipped). No CUDA/cuDF validation was performed.

This does not yet make public relational prediction fullgraph-compatible. Real pretrained RelBench driver-dnf prediction on 2.14 advances through the join and homogeneous graph construction, then fails at `TaskGraph.from_input`'s data-dependent `task_index.equal(arange)` validation. The same method also has data-dependent neighborhood traversal. The 2.7 public wrapper currently fails earlier under its context manager. Do not remove these validations or claim a whole-model fullgraph pass.


## Bounded task graph construction

A follow-up custom operator preserves the task-to-entity validation at runtime (including its ValueError), and exposes the validated readout length as the task-row count. Compiled traversal with an explicit hop count executes that many updates: once the frontier empties, remaining updates do nothing. Eager traversal and unbounded traversal retain their existing behavior. Unbounded traversal is not claimed fullgraph-compatible.

With these changes, actual CPU Inductor 2.14 public `predict` passes on pretrained RelBench driver-dnf: four query rows, one estimator, one captured graph / 5,714 captured calls; maximum absolute prediction difference 4.0531158447265625e-6 within unchanged atol=1e-5, rtol=1e-4. This is a compilation/correctness result, not a speed claim. The readout custom operator separately preserves valid, empty, duplicate-task, duplicate-entity, and missing-task behavior on both 2.7.1 and 2.14 (5/5 each).

Direct `TaskGraph.from_input` fullgraph Inductor checks also pass on both runtimes: 12 cases each, using hop counts 0/1/3 and populated, isolated, and multiple readouts. Inputs change between calls to the same compiled function. Readout indices and task assignments match eager exactly. This verifies bounded traversal after the frontier becomes empty; it does not validate unbounded traversal.

Additional compiled/eager contract checks pass on both runtimes: uint8 index overflow raises the same ValueError, floating index dtype raises the same TypeError, unequal join-key counts raise the same ArrowInvalid, and an explicit CPU output-device override preserves dtype and pairs. CUDA output-device overrides remain untested.
