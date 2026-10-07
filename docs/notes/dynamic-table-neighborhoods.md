# Dynamic table and relational neighborhood sizes

This branch removes three sources of unnecessary shape specialization. It is based on the preprocessing integration prototype, not standalone support on main.

A real KumoRelational prediction stream used ten 16-row query batches, six related tables, and two relationship hops. Different queries produce different neighborhood sizes. With `torch.compile(model.predict, fullgraph=True, dynamic=True)`, the previous prototype captured eight graphs and failed on the ninth query at the default recompilation limit. `dynamic=True` alone could not prevent these explicit specializations.

## Changes

1. Reconstruct `ColumnarTensor` dimensions from its tensor leaves. PyTorch passes concrete outer dimensions when recursively reconstructing nested wrappers, even when the leaves retain symbolic dimensions. Empty columnar blocks use their existing empty tensor; populated blocks use their first column. Columnar wrapper strides remain canonical, matching its constructor, while column views retain their own strides and storage.
2. Group ensemble tables by static schema, dtype, device, and category identities, then compare shapes within that group. Hashing a symbolic size forced it to the current integer. The new grouping preserves group/member order and incompatible shapes remain separate. Shape relationships may still require guards when grouping membership changes. Comparing multiple different shapes within one schema group is quadratic in the number of those groups; ordinary equal-shape members still require one comparison each.
3. Keep the native COO-to-CSR kernel behind a custom operator whose node-count argument is symbolic. The native operator declares a concrete `int size`, which specialized the sum of all neighborhood rows. The new operator declares `SymInt size` through PyTorch's schema inference and provides its output shape. Its runtime implementation calls the same native kernel. Eager graph construction keeps the original direct call.
4. Support runtime symbolic-size queries on `ColumnarTensor`, `StringTensor`/`VarLenTensor`, and `NullableTensor`. Dynamic captured graphs emit `aten.sym_size.int`; these wrappers previously rejected it. Categorical tensors already forward this operation to their codes.

The relevant operator schemas make the CSR issue explicit:

```text
aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor
sdm::_coo_to_csr(Tensor indices, SymInt size, *, bool out_int32=False) -> Tensor
```

The CSR fix is a separate commit. A `searchsorted` formulation confirmed the diagnosis but was removed because it would change the graph-construction algorithm and potentially its performance.

## Validation

| Check | PyTorch 2.7.1 | PyTorch 2.14 |
|---|---|---|
| Focused eager/compile regression suite | 49 passed, 5 skipped | 49 passed, 5 skipped |
| Forced-dynamic table input, nine distinct row counts, noncontiguous numerical columns, both graph policies | Passed | Passed |
| Independent ensemble grouping, eight shape/schema/dtype/member-count cases, both graph policies, actual CPU Inductor | Passed | Passed |
| Real ten-neighborhood prediction stream, fullgraph frontend capture | Not claimed | Two graphs; all ten queries execute |
| Same stream, actual CPU Inductor, fullgraph | Not claimed | Two graphs; all ten queries execute |

The frontend stream uses `backend="eager"` only to measure capture and guards. It is not an optimized-kernel performance result. Its existing strict prediction-tolerance failures remain: maximum absolute error is `6.914138793945312e-05`, unchanged from the preceding prototype. This change fixes shape support, not that separate numerical discrepancy. Actual CPU Inductor also completed all ten queries using two graphs at the default limit. All 160 predicted classes matched eager; three batches missed the existing strict probability tolerance, with maximum absolute error `6.246566772460938e-05`. The numerical discrepancy is therefore still a separate blocker to claiming strict prediction parity. No CUDA speed or memory improvement is claimed here.

CPU `opcheck` passed four dtype/empty-input combinations on each runtime; another 48 actual-Inductor calls per runtime matched an independent histogram/prefix-sum reference exactly, including empty edges with nonzero node counts. The portable probe is `experiments/compile_table_shapes/check_csr.py --device cpu` (use `--device cuda` for a GPU).

The focused regression tests cover eager ensemble grouping order, columnar size queries, table construction/views, changing dimensions, and exact CSR correctness against an independent reference for int32/int64 indices, empty inputs, and strided indices. Each compiler test resets Dynamo so independent parameterizations do not consume one another's recompilation limit.

Run the focused checks:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
python -m pytest -q test/test_ensemble.py test/tensor/test_columnar.py \
  test/tensor/test_table_compile.py \
  test/models/kumo/relational/test_graph_compile.py
```

The real-data command requires the existing driver-dnf validation bundle and KumoRelational classifier checkpoint; set `SDM_DATA` and `SDM_CHECKPOINT` to those local files:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 PYTHONPATH=. \
python experiments/compile_public_paths/probe.py \
  --model relational --entry predict --arm-index 1 --fullgraph \
  --capture-dynamic-outputs --backend eager \
  --relational-query-indices 3 4 5 6 7 8 9 10 11 3 \
  --data "$SDM_DATA" --checkpoint "$SDM_CHECKPOINT" \
  --output /tmp/relational-neighborhoods.json
```

The recorded frontend result is in `experiments/compile_table_shapes/results/relational-neighborhoods-frontend214.json`. The actual-Inductor summary is `relational-neighborhoods-inductor214.json` in the same directory, run against integration commit `6df9aeab5`; prediction arrays are omitted from that summary. `experiments/compile_table_shapes/check_grouping.py` reproduces the independent actual-Inductor grouping checks. Neither validation raises the recompilation limit or changes prediction tolerances. Different schemas, category identities, grouping relationships, empty dimensions, or model configuration may legitimately require another graph; this branch does not promise a single graph for every workload.

## Sparse-index stride correction

Fresh-process isolation showed that the native COO-to-CSR kernel ignores index strides on both CPU and CUDA. For logical indices `[0, 2, 4, 6, 8]`, it read the underlying contiguous prefix instead and returned incorrect pointers. The initial tests compared against the same native limitation, so their reference was insufficient; they now use independent histogram counts and prefix sums. CUDA `opcheck` also reported illegal memory access with the strided input, while direct fresh-process calls returned incorrect values.

The compiled wrapper now calls `indices.contiguous()` before the native kernel. The graph builder's sorted indices are already contiguous, so this is a no-op on its normal path. It does not change eager graph construction. CPU operator checks and all 48 dynamic cases per version pass against the corrected reference. CUDA checks are being repeated after this correction; do not treat the CPU results as CUDA validation.

The native implementation reads raw contiguous offsets rather than tensor strides: [PyTorch CUDA CSR kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/sparse/cuda/SparseCsrTensorMath.cu#L48-L75). `experiments/compile_table_shapes/isolate_csr.py` runs each native/wrapper, dtype, and layout combination in a fresh process.
