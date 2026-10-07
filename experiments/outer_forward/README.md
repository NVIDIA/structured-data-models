# Outer model compilation

This branch combines public preprocessing integration `17a797b3a`, native fitting arithmetic, atomic tensor construction, explicit fitted-buffer registration, and the categorical vocabulary-selection fix. It is a stacked investigation branch, not an independent patch against main.

The tested call is `torch.compile(model, fullgraph=flag, dynamic=True)(context, target, query, ...)`. This includes fitting a fresh preprocessing recipe and running uncached neural inference. It differs from compiling `model.predict` after eager fitting.

## CPU results

PyTorch 2.14.0, actual Inductor, pretrained KumoTabular small, FP32, one estimator, 32 context rows and four query rows. Eager and compiled runs receive generators seeded identically. The existing probe uses atol=1e-5, rtol=1e-4; no tolerances were changed.

| Task | Graph breaks allowed | Fullgraph |
|---|---|---|
| Breast-cancer classification | Pass; max absolute error 7.7486e-7 | Fails when category shuffling branches on the learned vocabulary size |
| Diabetes regression | Pass; max absolute error 0.000366211 | Fails tracing Bernoulli sampling with an explicit torch.Generator |

Partial compilation can hit the default processor recompilation limit and run affected regions eagerly. A passing result does not mean all Python setup or every tensor operation was compiled. Larger query sets and GPU execution are not established by these four cases.

Full errors and settings are preserved in [results214.json](results214.json). The shared harness is [probe.py](../compile_public_paths/probe.py). Reproduce with:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
python experiments/compile_public_paths/probe.py \
  --model tabular --entry forward --task classification \
  --data /path/to/classification/data_train_validation.npz \
  --checkpoint /path/to/small/classifier.pt --output result.json
```

Add `--fullgraph` for the strict case. Use `--task regression`, the diabetes dataset and `regressor.pt` for regression. The datasets are the existing train/validation-only audit artifacts described by the public integration experiments.

## Remaining work

Keep explicit random-generator state and learned vocabulary semantics. Dropping the generator or fitting all categories regardless of observed data would change behavior and is not a fix. Preparing random choices outside a compiled tensor function is a possible API boundary; it would not make the entire public fitting call a single graph.

The same four outer-forward cases were also run on actual CPU PyTorch 2.7.1 Inductor. Both tasks fail partial compilation while resuming construction of a TaskDispatch module (`object has no attribute '_modules'`). Both strict cases reject construction of a frozenset from a generator during recipe setup. See [results27.json](results27.json). Passing an explicitly prepared recipe is a separate experiment on the fitting branch; these results use the default public call unchanged.

No GPU or end-to-end speed claim is made here.

## Fresh-cache follow-up on 2.7

Passing an eagerly constructed recipe via `--prepared-recipe` gets past default recipe setup. At source `9879a9983`, actual Inductor outer classification then passes with graph breaks allowed, maximum error 5.96e-8. This includes fitting and the uncached internal model call.

A controlled comparison uses distinct fresh cache directories, with FX/AOT caches disabled: parent `2a428e0ac` fails by routing a dtype argument to `unsqueeze`; adding the atomic table-dispatch boundary passes. See [fresh-cache27.json](fresh-cache27.json) for exact environments, commands and results. The boundary preserves compiler access to tensor computation; this is still partial compilation.

Earlier `expected 13` argument-count errors in [prepared-recipe27.json](prepared-recipe27.json) came from incompatible cached generated code after experimental tensor flatten layouts changed. They are not evidence that column shuffling needs a source fix. Fresh caches avoid that artifact; compatibility of old compiled artifacts across wrapper-layout changes remains separate work.
