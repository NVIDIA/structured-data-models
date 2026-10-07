# Compiling default recipe preprocessing

This branch fixes three tracing failures in recipe and ensemble orchestration. It does not change numerical algorithms, skip preprocessing, or disable compilation. Base: `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`.

## Changes and failing code

| Before | Observed failure | Change |
|---|---|---|
| `any(isinstance(m, sp.TaskDispatch) for m in self.target.modules())` during recipe construction | PyTorch 2.14: `getattr() on nn.Module with pending mutation` in recursive `named_modules()` | Traverse registered children directly, retaining duplicate/cycle handling and the same route validation. |
| `group.active_stypes & self.handles_stypes` | PyTorch 2.14: `SourcelessBuilder.create does not know how to wrap <class 'frozenset'>` | Check the existing column schema for a handled, nonempty semantic type. |
| `for stype, block in table.items()` while repacking ensemble members | PyTorch 2.14: `torch.* op returned non-Tensor`, generator returned by `items` | Call `TableTensor.items(table)` to inline the existing Python iterator rather than treating it as a tensor operation. |

Moving validation before assignment to `self._target` was also tried; recursive PyTorch `named_modules()` still failed on newly constructed children. The traversal change preserves validation rather than bypassing it.

## Dependency scope

The recipe-construction fix works independently. Compiling actual table preprocessing additionally requires the container and processor applicability fixes developed in parallel. This focused branch does **not** duplicate those source changes.

The validated integration applied these commits in addition to this branch:

```text
d75aa7bd6  CategoricalTensor flatten/unflatten
9396a9099  TableTensor/ColumnarTensor flatten/unflatten
db66716a1  TableTensor public block tracing
11cbacca0  CategoricalTensor public code tracing
50f86beec  Symbolic ColumnarTensor reconstruction and table aliases
a18d41b19  Categorical aliases and tracing-safe representation
8a2569fe5  Empty variable-length payload concatenation
276cc5a6a  Processor semantic-type applicability
8041c50c1  Processor applicability from column metadata
```

These are prototype dependencies, not a claim that all container changes are ready to merge.

## What was validated

CPU macOS; PyTorch 2.7.1/Python 3.12 and PyTorch 2.14.0/Python 3.13. The successful compilation checks below use actual Inductor, not the `eager` capture backend.

`validate.py` eagerly fits each default recipe on the first 300 breast-cancer dataset rows, then directly compiles `RecipeExecution.transform`. The input has 30 real numerical features plus one constant column. It checks held-out query batches of 31, 47, and 31 rows, preserves column metadata, checks output parity, and verifies every fitted preprocessing buffer remains unchanged. The constant column is removed. All four KumoTabular preprocessing choices are exercised with four ensemble members.

| Scope | 2.14 result |
|---|---|
| Construct either default recipe within a full graph | Pass |
| KumoTabular recipe, one member, `fullgraph=False` | Pass; exact eager output |
| Both recipes, one member, `fullgraph=True`, static and dynamic compilation | Pass; exact eager output across changing query row counts |
| KumoTabular recipe, four members, `fullgraph=True`, static and dynamic compilation | Pass; exact eager output |
| KumoRelational task-table recipe, two members, `fullgraph=True`, `dynamic=True` | Pass; exact eager output |
| Existing recipe, ensemble processor, and semantic-type routing tests | 26 passed, one CUDA case skipped on **each** runtime |

The final dynamic runs each captured two graphs across 31 → 47 → 31 query rows; the second row count triggered a shape guard. The same happened for 38 → 47 → 38. Forcing dynamic rows with `mark_dynamic` exposes specialization during table reconstruction/shape consistency checks. Dynamic mode preserves correctness here but does not yet provide graph reuse across changing row counts. An alternative ensemble shape-grouping implementation did not fix this and was discarded.

The Relational fixture uses task-table numerical features. It does not exercise related-table datetime features, Arrow joins, categorical dictionaries, the internal neural model, or CUDA. No inference-speed or memory claim is made.

## Commands

After applying the dependency commits to an isolated checkout, use a Python environment with SDM dependencies, PyTorch, and scikit-learn:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/recipe-compilation/validate.py \
  --model tabular --members 4 --fullgraph --dynamic

PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/recipe-compilation/validate.py \
  --model relational --members 2 --fullgraph --dynamic

PYTHONPATH=. OMP_NUM_THREADS=1 python experiments/recipe-compilation/probe.py \
  --model tabular --mode construct --backend inductor --fullgraph

PYTHONPATH=. OMP_NUM_THREADS=1 python experiments/recipe-compilation/probe.py \
  --model tabular --mode fit_transform --fullgraph

PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest \
  test/processing/test_recipe.py test/processing/test_ensemble.py \
  test/processing/common/test_stype.py -q
```

The precompiled-header environment setting works around a local macOS compiler issue. It does not alter tensor arithmetic. `validate.py --data PATH` can load a locally exported `(features, targets)` tensor tuple instead of importing scikit-learn; this was used for the 2.7 environment.

## Remaining blockers

- **PyTorch 2.7.1:** Default full-graph recipe construction still rejects `frozenset(generator)`. Preconstructed recipes also encounter unsupported `frozenset` module attributes; allowing graph breaks can instead reach an assertion in Dynamo's `DictGetItemSource` for semantic-type enum keys. A tuple-materialization workaround only moved the first failure and was discarded. These results do not establish 2.7 preprocessing compilation support.
- **Compiled fitting:** The actual default recipe reaches `DropConstantColumns._keep_mask(...).tolist()`, which determines a Python column schema from training values. Casting the mask to integers merely moves the blocker to data-dependent Python decisions. Supporting this requires an explicit schema-learning boundary or a larger tensorized schema design; silently retaining constant columns would change behavior.
- **Categorical transform:** The `validate.py --categorical` fixture turns the first three real features into three-valued category codes and keeps the remaining numerical features. Full-graph 2.14 tracing currently fails when `AlignCategories` reads `table.categorical.categories`: the compiler cannot wrap its returned tuple of fake tensors. This is separate from numerical recipe support.
- **Other inputs:** Text/date columns, actual related tables, and CUDA need their own validation. Working numerical recipe transforms are useful progress, not complete public `fit`/`predict` support.
