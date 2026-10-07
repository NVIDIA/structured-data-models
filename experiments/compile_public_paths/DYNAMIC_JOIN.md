# Variable-output join compilation prototype

The identifier-key path packages ordinary, nullable, and string column storage as tensor leaves and invokes the existing Arrow/cuDF join through a custom operator. The fake implementation gives both returned index tensors the same dynamic length. Arrow/cuDF remains external code; the compiler does not optimize its join algorithm. Other semantic column types retain the existing eager boundary.

Use a scoped `torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)` around compilation and execution. This option is necessary because duplicate keys, missing keys, and empty inputs change the number of result pairs without changing the input shapes.

Run `PYTHONPATH=. python experiments/compile_public_paths/dynamic_join.py`. CPU Inductor with `fullgraph=True, dynamic=True` passes 24 comparisons on each of PyTorch 2.7.1 and 2.14: duplicate keys, empty left/right, no matches, NaN and numeric casting, nullable IDs, string/null IDs, and multi-column keys, each with int64/int32/uint8 indices. Pair order and dtype match the existing join exactly in those cases. Existing join tests pass (6 passed, 5 CUDA skipped). No CUDA/cuDF validation was performed.

This does not yet make public relational prediction fullgraph-compatible. Real pretrained RelBench driver-dnf prediction on 2.14 advances through the join and homogeneous graph construction, then fails at `TaskGraph.from_input`'s data-dependent `task_index.equal(arange)` validation. The same method also has data-dependent neighborhood traversal. The 2.7 public wrapper currently fails earlier under its context manager. Do not remove these validations or claim a whole-model fullgraph pass.
