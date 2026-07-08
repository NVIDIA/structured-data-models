# TabFM to SDM integration log

## Purpose

This is the living implementation and validation record for adding TabFM to
the `structured-data-models` package. Update it in the same change as every
TabFM source, test, checkpoint, processing, or public API modification.

The implementation procedure is documented in
[`TABFM_SDM_INTEGRATION_PLAN.md`](TABFM_SDM_INTEGRATION_PLAN.md). Architectural
correspondences and compatibility boundaries are documented in
[`TABICLV2_TABFM_COMPARISON.md`](TABICLV2_TABFM_COMPARISON.md).

## Status summary

| Item                             | Status                                                  |
| -------------------------------- | ------------------------------------------------------- |
| Development branch               | `add-tabfm` created                                     |
| Upstream reference               | Cloned and pinned                                       |
| Upstream smoke baseline          | Passing                                                 |
| SDM package scaffold             | Added                                                   |
| Attention primitives             | Complete; upstream parity passing                       |
| Cell and column embeddings       | Complete; upstream parity passing                       |
| Row interaction                  | Complete; upstream parity passing                       |
| Dataset-wise ICL                 | Complete; upstream parity passing                       |
| Full SDM TabFM model             | Core and public wrapper complete; parity passing        |
| Official checkpoint loading      | Strict local loading complete; remote download disabled |
| SDM preprocessing recipe         | Single-estimator numeric recipes complete               |
| Context caching                  | Projected column and ICL context caching complete       |
| `torch.compile` coverage         | Full-graph eager-backend coverage complete              |
| Official checkpoint validation   | Harness complete; licensed weights pending              |
| CUDA validation                  | Harness complete; CUDA execution pending                |
| Package documentation            | TabFM guide complete; warnings-as-errors build passing  |
| Shared SDM neural utilities      | SDPA kernel and rotary application reused               |
| Public `sdm.models.TabFM` export | Complete                                                |

## Fixed upstream reference

| Field          | Value                                                       |
| -------------- | ----------------------------------------------------------- |
| Repository     | `https://github.com/google-research/tabfm.git`              |
| Local checkout | `/home/ruthvikak/tabfm-reference`                           |
| Pinned commit  | `633cd265f498e1d20c9625be0639f6305d8e2541`                  |
| Git state      | Detached HEAD at the pinned commit                          |
| Intended use   | Read-only parity oracle; not an SDM dependency or submodule |

Licensing notes:

- TabFM source is Apache 2.0. Preserve applicable copyright and license
  notices in adapted source and mark modified files.
- Released `google/tabfm-1.0.0-pytorch` weights use the TabFM Non-Commercial
  License v1.0. Do not redistribute them or enable an automatic download flow
  without maintainer approval of the license-handling policy.

## Change log

### 2026-07-07 — Integration planning and architecture analysis

Added:

- [`TABICLV2_TABFM_COMPARISON.md`](TABICLV2_TABFM_COMPARISON.md)
- [`TABFM_SDM_INTEGRATION_PLAN.md`](TABFM_SDM_INTEGRATION_PLAN.md)

Recorded:

- function- and strategy-level correspondence between TabICLv2 and TabFM;
- checkpoint-safe, parity-sensitive, and retraining-required substitutions;
- the recommended package structure and twelve-stage implementation sequence;
- source and checkpoint licensing constraints; and
- required tests and completion criteria.

Validation:

- Markdown whitespace validation passed with `git diff --check` or equivalent
  no-index checks.
- Local Markdown link targets were checked and found to exist.
- Repository `pre-commit` was not available in the original base environment at
  this point.

### 2026-07-07 — Pinned upstream baseline and SDM scaffold

Repository setup:

- created SDM branch `add-tabfm`;
- cloned TabFM to `/home/ruthvikak/tabfm-reference`;
- checked out commit `633cd265f498e1d20c9625be0639f6305d8e2541`;
- kept the reference checkout outside the SDM repository; and
- confirmed the reference checkout has no local modifications.

Added:

- [`sdm/models/tabfm/__init__.py`](sdm/models/tabfm/__init__.py)
- [`test/models/tabfm/test_upstream_reference.py`](test/models/tabfm/test_upstream_reference.py)

The package initializer intentionally exports no model yet. The public
`sdm.models.TabFM` export will be added only when a usable SDM implementation
exists.

The upstream smoke test loads `tabfm/src/pytorch/model.py` directly by file
location rather than importing the upstream top-level `tabfm` package. This
avoids pulling sklearn-wrapper dependencies such as `absl`, pandas, scipy, and
scikit-learn into the core-model baseline.

The smoke test covers:

- tiny classification and regression models;
- deterministic repeat execution;
- expected classification and regression output widths;
- finite outputs;
- numerical and categorical feature routing;
- padded feature widths through `d`; and
- different context lengths across batch members.

Test environment:

| Component   | Value                                                    |
| ----------- | -------------------------------------------------------- |
| Environment | Temporary virtual environment at `/tmp/tabfm-smoke-venv` |
| Python      | 3.12.3                                                   |
| PyTorch     | 2.12.1+cpu                                               |
| pytest      | 9.1.1                                                    |
| Device      | CPU                                                      |

The temporary environment was necessary because the base system Python had no
PyTorch, pytest, pip module, or repository-managed virtual environment.

Commands and results:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest \
  test/models/tabfm/test_upstream_reference.py -vv
```

```text
collected 2 items
test_upstream_tiny_forward_is_deterministic[True-3]  PASSED
test_upstream_tiny_forward_is_deterministic[False-1] PASSED
2 passed in 1.37s
```

Lint and format validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm/__init__.py \
  test/models/tabfm/test_upstream_reference.py

/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm/__init__.py \
  test/models/tabfm/test_upstream_reference.py
```

```text
All checks passed!
2 files already formatted
```

An initial baseline attempt failed because importing
`tabfm.src.pytorch.model` first executed upstream `tabfm/__init__.py`, which
imports `absl` through the sklearn estimator layer. The test was corrected to
load the standalone PyTorch source with `importlib.util.spec_from_file_location`.
The direct-load version passed without installing the upstream wrapper
dependencies.

### 2026-07-07 — Checkpoint-compatible attention primitives

Added:

- [`sdm/models/tabfm/attention.py`](sdm/models/tabfm/attention.py)
- [`test/models/tabfm/conftest.py`](test/models/tabfm/conftest.py)
- [`test/models/tabfm/test_attention.py`](test/models/tabfm/test_attention.py)

Refactored:

- [`test/models/tabfm/test_upstream_reference.py`](test/models/tabfm/test_upstream_reference.py)
  now obtains the directly loaded upstream module from the shared session
  fixture in `conftest.py`.

Implemented these checkpoint-compatible primitives in order:

1. `RMSNorm`
2. `RotaryEmbedding`
3. `MultiheadAttention`
4. `MultiheadAttentionBlock`
5. `Encoder`

Preserved upstream behavior and state-dict names for:

- float32 RMSNorm accumulation and output dtype restoration;
- checkpoint-loadable RoPE frequency buffers;
- interleaved rotary positions;
- separate q, k, v, and output projections;
- per-head q/k RMS normalization;
- learned positive per-dimension query scaling;
- SDPA with explicit scale `1.0`;
- TabFM residual and normalization order;
- SwiGLU and non-gated GELU feed-forward paths;
- shared encoder-level RoPE; and
- optional feed-forward activation chunking through `ffn_chunk_size`.

Added input validation for invalid channel/head dimensions, RoPE head width,
theta, epsilon, feed-forward width, mismatched q/k/v shapes, and unsupported
activations. Public classes and methods use typed boundaries and document tensor
shapes, dtype behavior, and constructor parameters.

Direct parity coverage includes:

- RMSNorm;
- RoPE;
- masked and unmasked cross-attention;
- attention with and without RoPE;
- SwiGLU and GELU attention blocks;
- chunked versus unchunked feed-forward evaluation;
- encoders with and without RoPE;
- float32 CPU; and
- bfloat16 CPU encoder execution.

Focused command:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest test/models/tabfm -q
```

Final focused result:

```text
15 passed in 0.69s
```

Repository-wide command:

```bash
/tmp/tabfm-smoke-venv/bin/pip install -e '.[test]'
/tmp/tabfm-smoke-venv/bin/pytest
```

Repository-wide result:

```text
205 passed, 81 skipped in 7.02s
```

The skipped cases require unavailable CUDA, CuPy, or cuDF support. There were
no test failures.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm/attention.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm/attention.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
git diff --check
```

Final result: all Ruff, format, type, and whitespace checks passed.

Issues found and resolved:

1. The first focused run could not import `sdm` because the temporary
   environment did not contain the project. Adding only `PYTHONPATH` then
   exposed the missing core `pyarrow` dependency. The project and test extras
   were installed editable into the isolated environment.
2. A zero-tolerance chunking assertion failed with a maximum float32 absolute
   difference of `4.76837158203125e-07`. Different GEMM batch sizes can change
   rounding without changing the computation. The test now uses the same tight
   `rtol=1e-5`, `atol=1e-6` parity tolerance as the other float32 components.
3. Initial Ruff validation found one long line and one import-order issue in the
   tests. Both were corrected; final Ruff and format validation passed.
4. The first direct `ty` invocation searched the system Python rather than the
   isolated environment and could not resolve PyTorch or pytest. Supplying
   `--python /tmp/tabfm-smoke-venv/bin/python` resolved the environment and
   identified the RoPE buffer as an ambiguous `Tensor | Module`. Adding the
   explicit `freqs: Tensor` class annotation resolved the diagnostic.
5. `pre-commit run --files ...` could not create its hook environment because
   `.pre-commit-config.yaml` requires Python 3.10 and this machine only exposes
   Python 3.12. Equivalent Ruff, format, `ty`, whitespace, focused-test, and
   repository-test checks were run directly and passed.

### 2026-07-07 — Induced attention and column embedding

Added:

- [`sdm/models/tabfm/embedding.py`](sdm/models/tabfm/embedding.py)
- [`test/models/tabfm/test_embedding.py`](test/models/tabfm/test_embedding.py)

Implemented:

1. `InducedSelfAttentionBlock`
2. `SetTransformer`
3. `ColEmbedding`

Preserved upstream topology, state-dict names, and behavior for:

- `ind_vectors`, `mab1`, and `mab2` parameter/module names;
- inducing vectors attending to context rows before rows attend back to the
  inducing representation;
- stacked induced self-attention blocks;
- folding columns into the batch axis as `[B * H, T, D]`;
- per-table context masks expanded across columns;
- the column output projection and RMSNorm;
- restoration to `[B, T, H, D]`; and
- optional independent-column chunking through `col_chunk_size`.

The implementation adds typed device/dtype construction, input shape and dtype
validation, GPU-safe tensor operations, shape documentation, the Set
Transformer citation, and retained Apache attribution with a modification
notice.

Added parity and behavior tests for:

- induced attention with and without a context mask;
- a two-block set transformer with and without a context mask;
- full column embedding in float32 and bfloat16;
- chunked and unchunked column execution;
- invalid `train_size` shape and floating dtype; and
- query-row isolation: perturbing one test row changes that row but does not
  change other test rows or another table.

Focused command:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest \
  test/models/tabfm/test_embedding.py \
  test/models/tabfm/test_attention.py \
  test/models/tabfm/test_upstream_reference.py -vv
```

Focused result:

```text
26 passed in 0.83s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest
```

```text
216 passed, 81 skipped in 5.95s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing SDM test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm/embedding.py test/models/tabfm/test_embedding.py
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm/embedding.py test/models/tabfm/test_embedding.py
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
```

Final result: all lint, format, and type checks passed.

Issues found and resolved:

1. Initial Ruff validation rejected a split first docstring sentence because it
   did not see the required blank line after the summary. The citation was
   changed to a short named reStructuredText link, preserving a one-line
   summary followed by a blank line.
2. Ruff found one import-order issue in `test_embedding.py`; its automatic
   import organization was applied and the final check passed.

### 2026-07-07 — MLP and Fourier cell embedding

Added:

- [`sdm/models/tabfm/_utils.py`](sdm/models/tabfm/_utils.py)
- [`sdm/models/tabfm/mlp.py`](sdm/models/tabfm/mlp.py)

Extended:

- [`sdm/models/tabfm/embedding.py`](sdm/models/tabfm/embedding.py)
- [`test/models/tabfm/test_embedding.py`](test/models/tabfm/test_embedding.py)

Refactored:

- moved the checkpoint-compatible tanh-GELU and activation lookup from
  `attention.py` into the private `_utils.py` module;
- reused the same activation implementation from both
  `MultiheadAttentionBlock` and `MLP`; and
- preserved all attention parity after the refactor.

Implemented `MLP` with upstream-compatible `layers.N` state-dict names,
activation placement between hidden layers only, and support for ReLU, tanh
GELU, and SiLU.

Implemented `CellEmbedder` with upstream-compatible names and behavior for:

- cyclic feature-group offsets `(2**index) - 1`, yielding `[0, 1, 3]` for the
  released group size;
- vectorized grouping without a Python feature loop;
- per-table wrap-around using active feature count `d`;
- separate numerical and categorical Fourier-frequency buffers;
- separate numerical and categorical linear projections;
- float32 Fourier angle and sine/cosine evaluation;
- casting Fourier features back to model compute dtype before projection;
- summation across feature-group slots;
- categorical routing at the individual group-slot level;
- classification target lookup and regression target MLP;
- context-only target injection using per-table `train_size`;
- zeroing output columns at or after each table's active feature count; and
- optional independent-row chunking through `row_chunk_size`.

The implementation retains upstream state names including
`fourier_frequencies`, `fourier_frequencies_cat`, `in_linear`,
`in_linear_cat`, and `y_embedder_lookup`. It adds typed device/dtype
construction, tensor shape documentation, validation of input metadata, Apache
attribution, and a modification notice.

Parity tests randomize both Fourier-frequency banks before copying the upstream
state. This ensures the tests exercise nontrivial Fourier angles instead of
passing only through the zero-initialized frequency placeholder.

Added tests for:

- MLP parity for ReLU, GELU, and SiLU in float32 and bfloat16;
- exact `[0, 1, 3]` feature-group indices;
- active-width wrap-around with padded physical feature width;
- classification and regression cell embedding;
- float32 and bfloat16 cell embedding;
- categorical routing enabled and disabled;
- active feature counts enabled and disabled;
- row chunking enabled and disabled;
- padded output columns being exactly zero;
- query/test target changes having no effect on outputs; and
- context-target changes affecting only the corresponding context row at this
  independent cell-embedding stage.

Focused command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest test/models/tabfm -q
```

```text
68 passed in 0.93s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest
```

```text
258 passed, 81 skipped in 5.86s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
git diff --check
```

Final result: all lint, format, type, and whitespace checks passed. All newly
added parity and behavior tests passed on their first complete run; no model
implementation correction was required in this stage.

### 2026-07-07 — Row interaction

Added:

- [`sdm/models/tabfm/row_interaction.py`](sdm/models/tabfm/row_interaction.py)
- [`test/models/tabfm/test_row_interaction.py`](test/models/tabfm/test_row_interaction.py)

Implemented `RowInteraction` with upstream-compatible state and module names:

- `tf_row` for the RoPE-enabled encoder;
- `out_ln` for output RMSNorm;
- `num_cls` and `output_full` behavior flags; and
- `row_chunk_size` for independent-row activation chunking.

Preserved upstream behavior for:

- folding `[B, T, H, D]` into `[B * T, H, D]` so rows are processed
  independently;
- feature-position RoPE with base 100,000;
- full-sequence output for the first row stage;
- leading-CLS selection, normalization, and flattening for the second row
  stage;
- active-feature masks that admit `num_cls + d` key/value tokens;
- returning `[B, T, H, D]` in full mode;
- returning `[B, T, num_cls * D]` in CLS-only mode; and
- exact independent-row chunking through `row_chunk_size`.

Added typed device/dtype construction, input and active-width validation,
public tensor shape documentation, retained Apache attribution, and a
modification notice.

Added tests for all combinations of:

- full-sequence and CLS-only output;
- float32 and bfloat16 execution;
- active-feature masking enabled and disabled; and
- row chunking enabled and disabled.

Additional behavior tests verify:

- perturbing masked padded feature tokens cannot affect CLS or valid-feature
  outputs;
- padded query outputs can still change in full-output mode because only
  keys/values are masked, matching upstream behavior;
- perturbing one table row cannot affect any other row or table; and
- invalid active-feature shape and floating dtype are rejected.

Dedicated row-interaction result:

```text
22 passed in 0.86s
```

Focused TabFM command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest test/models/tabfm -q
```

```text
90 passed in 1.21s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest
```

```text
280 passed, 81 skipped in 6.12s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm/row_interaction.py \
  test/models/tabfm/test_row_interaction.py
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm/row_interaction.py \
  test/models/tabfm/test_row_interaction.py
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
```

Final result: all lint, format, and type checks passed.

Issue found and resolved:

- Initial Ruff validation found one import-order issue in
  `test_row_interaction.py`. Ruff's import organization was applied, after
  which lint, format, type, focused, and full-suite validation passed.

### 2026-07-07 — Dataset-wise in-context learning

Added:

- [`sdm/models/tabfm/icl.py`](sdm/models/tabfm/icl.py)
- [`test/models/tabfm/test_icl.py`](test/models/tabfm/test_icl.py)

Implemented:

1. `OneHotAndLinear`
2. `ICLearning`

`OneHotAndLinear` preserves upstream state and behavior:

- `projection` retains the released checkpoint parameter name;
- valid class IDs are one-hot encoded and linearly projected;
- invalid, negative, and padded class IDs map to an all-zero one-hot vector;
  and
- invalid-class output therefore equals the projection bias exactly.

`ICLearning` preserves upstream state names and behavior for:

- `tf_icl`, `ln`, `y_encoder`, and `decoder` modules;
- RoPE-disabled attention across table rows;
- one-hot classification target projection;
- regression target MLP encoding;
- context-only target addition using per-table `train_size`;
- context-only attention keys and values;
- full-row query evaluation;
- final RMSNorm;
- classification decoding to `max_classes` logits; and
- regression decoding to one scalar per row.

Added typed device/dtype construction, tensor shape documentation, input and
metadata validation, retained Apache attribution, and a modification notice.

Added tests for:

- `OneHotAndLinear` float32 and bfloat16 upstream parity;
- valid, negative, boundary, and out-of-range class IDs;
- exact projection-bias output for invalid classes;
- classification and regression ICL parity;
- float32 and bfloat16 ICL parity;
- classification and regression output shapes;
- query/test targets having no effect on any prediction;
- context-target changes affecting query predictions;
- one table's context changes not affecting another table;
- perturbing one query-row representation not affecting other query rows; and
- invalid `train_size` shape and floating dtype rejection.

Dedicated ICL result:

```text
13 passed in 0.67s
```

Focused TabFM command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest test/models/tabfm -q
```

```text
103 passed in 1.28s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q
```

```text
293 passed, 81 skipped in 6.28s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm/icl.py test/models/tabfm/test_icl.py
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm/icl.py test/models/tabfm/test_icl.py
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
```

Final result: all lint, format, and type checks passed.

Issues found and resolved:

1. Initial Ruff validation found one import-order issue in `test_icl.py`; Ruff's
   import organization was applied.
2. Ruff format validation requested routine formatting in `icl.py`; the file
   was formatted and all subsequent lint, format, type, focused, and full-suite
   checks passed.

### 2026-07-07 — Checkpoint-compatible TabFM neural core

Added:

- [`sdm/models/tabfm/model.py`](sdm/models/tabfm/model.py)
- [`test/models/tabfm/test_model.py`](test/models/tabfm/test_model.py)

Implemented `TabFMCore` with the upstream constructor/configuration keys and
checkpoint child-module names:

- `cell_embedder`;
- `col_embedder` and `col_embedder_2`;
- `row_interactor` and `row_interactor_2`;
- `cls_tokens`; and
- `icl_predictor`.

The forward path preserves the released model's stage order exactly:

```text
NaN sentinel conversion and compute-dtype cast
  -> CellEmbedder
  -> first ColEmbedding
  -> prepend learned CLS tokens
  -> RowInteraction with full output
  -> second ColEmbedding
  -> RowInteraction with CLS-only output
  -> ICLearning
```

The core also preserves the upstream default activation chunk sizes:

- 4,096 rows for the cell embedder and both row-interaction stages;
- 16 feature instances for both column stages; and
- 8,192 tokens for every transformer feed-forward block.

The SDM adaptation adds typed device/dtype construction, validates a positive
feed-forward multiplier, documents the public constructor and tensor shapes,
and retains the upstream Apache 2.0 notice plus a modification notice. It does
not yet add the SDM-facing query-row adapter or public export.

Added end-to-end tests for:

- strict, zero-remap loading of the pinned upstream model state dictionary;
- classification and regression;
- float32 and bfloat16 computation;
- numerical-only and categorical-routing inputs;
- active feature widths through `d`;
- unchunked and deliberately small forced-chunk execution;
- entry-point NaN replacement with the `-100` sentinel;
- output widths for both tasks;
- exact released chunk defaults; and
- query/test targets having no effect on any prediction.

Dedicated core result:

```text
11 passed in 1.29s
```

Focused TabFM command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q test/models/tabfm
```

```text
114 passed in 2.08s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q
```

```text
304 passed, 81 skipped in 7.12s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
git diff --check
```

Final result: all lint, format, type, and whitespace checks passed.

Issues found and resolved:

1. The first Ruff run found three 80-character docstring lines, an unused
   unpacked shape channel, and unsorted imports in the new test. The docstrings
   and binding were corrected, and Ruff organized the test imports.
2. The first `ty` invocation omitted `--python`, so `ty` searched only the base
   environment and reported 21 unresolved PyTorch imports. Re-running with the
   isolated test interpreter resolved those environment diagnostics.
3. After import cleanup, `ty` found an imprecisely inferred heterogeneous
   configuration dictionary and dynamic chunk attributes in the upstream test
   helper. The configuration was explicitly typed and the three intentional
   dynamic assignments received narrow `ty` ignores. The final type check
   passed with no diagnostics.

### 2026-07-08 — Public SDM wrapper and exports

Changed:

- [`sdm/models/tabfm/model.py`](sdm/models/tabfm/model.py)
- [`sdm/models/tabfm/__init__.py`](sdm/models/tabfm/__init__.py)
- [`sdm/models/__init__.py`](sdm/models/__init__.py)
- [`test/models/tabfm/test_model.py`](test/models/tabfm/test_model.py)

Added the public `TabFM` adapter as a thin `BaseModel` subclass around separate
classification and regression `TabFMCore` instances. The wrapper:

- selects classification for integer targets and regression for
  floating-point targets;
- accepts direct core composition through `cls_model` and `reg_model`;
- supports unbatched inputs and arbitrary leading batch dimensions;
- infers a uniform context size from `y` or accepts per-table `train_size`;
- returns query rows only;
- left-packs variable-length query predictions and zero-pads shorter query
  tails;
- carries explicit `cat_mask` and active feature counts `d` to the core;
- preserves numerical and categorical `TableTensor` columns instead of
  dropping categorical data;
- converts categorical indices to the model input dtype and infers their mask;
- rejects unsupported datetime and identifier `TableTensor` columns rather
  than silently ignoring them;
- supports `fit`/`predict` parity by storing prepared raw context; and
- validates replay batch shape, feature width, categorical routing, and active
  widths.

The current `fit`/`predict` implementation deliberately stores raw prepared
context and recomputes the full forward pass. It does not claim transformer
key/value caching; that optimization remains a later parity-gated step.

Added the public exports:

```python
from sdm.models import TabFM
from sdm.models.tabfm import TabFM, TabFMCore
```

`pretrained=False` is the current default and performs no download.
`pretrained=True` raises a focused `NotImplementedError` explaining that the
released non-commercial weights require an approved package license policy.
The default recipe is temporarily an identity `Recipe`; tensor-native TabFM
preprocessing remains a separate step.

Added wrapper tests for:

- classification and regression task routing;
- unbatched, one-dimensional batch, and multi-dimensional batch shapes;
- equality between flattened batched execution and per-batch execution;
- variable per-table context lengths and padded query output;
- exact `TableTensor` numerical/categorical routing parity with explicit tensor
  input and `cat_mask`;
- `fit`/`predict` equality with direct `forward` for both tasks;
- categorical-mask and active-width replay;
- clearing fitted context;
- public package exports;
- constructor representation;
- the license-safe `pretrained=True` failure; and
- rejection of task-incompatible custom cores.

Dedicated model test result:

```text
22 passed in 1.77s (final rerun after adding TableTensor replay coverage)
```

Focused TabFM command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q test/models/tabfm
```

```text
125 passed in 2.36s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q
```

```text
315 passed, 81 skipped in 7.33s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm sdm/models/__init__.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm sdm/models/__init__.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm sdm/models/__init__.py test/models/tabfm
git diff --check
```

Final result: all lint, format, type, and whitespace checks passed.

Issues found and resolved:

1. The first type check rejected a tensor scalar passed directly as the end of
   `torch.arange`, although PyTorch supports it at runtime. Query packing was
   rewritten as a fixed integer range followed by a tensor mask, retaining a
   graph-friendly tensor formulation and satisfying the typed API.
2. The first dedicated wrapper run reported `21 passed, 1 failed`. The model
   output was correct; the variable-context test incorrectly indexed an
   unbatched result and compared `[2, K]` against `[K]`. Removing that extra
   test index produced the intended per-table comparison.
3. Ruff format requested routine formatting for the expanded `model.py`.
   Formatting was applied and the subsequent format check passed.
4. `ty` inferred the wrapper test's heterogeneous configuration dictionary too
   narrowly and reported four constructor diagnostics. Annotating it as
   `dict[str, Any]` resolved the inference issue; the final type check passed.

### 2026-07-08 — License-safe local checkpoint loading

Added and changed:

- [`sdm/models/tabfm/checkpoint.py`](sdm/models/tabfm/checkpoint.py)
- [`sdm/models/tabfm/model.py`](sdm/models/tabfm/model.py)
- [`test/models/tabfm/test_checkpoint.py`](test/models/tabfm/test_checkpoint.py)
- [`test/models/tabfm/test_model.py`](test/models/tabfm/test_model.py)
- [`pyproject.toml`](pyproject.toml)

Implemented explicit local checkpoint loading through:

```python
TabFM(pretrained=True, checkpoint_path="/path/to/local/tabfm-checkpoint")
```

The expected upstream-compatible layout is:

```text
checkpoint_path/
  classification/
    config.json
    model.safetensors
  regression/
    config.json
    model.safetensors
```

`pytorch_model.bin` is supported as a fallback in either variant directory to
match the fallback handled by the upstream Hugging Face mixin. SafeTensors is
preferred whenever both files exist. Added `safetensors` as a direct package
dependency because the official PyTorch release uses `model.safetensors`.

The loader:

- requires both classification and regression variants;
- constructs each `TabFMCore` from its own `config.json`;
- maps the official `task` key to `is_classifier`;
- ignores only the upstream metadata keys `framework`, `model_type`, and
  `version`;
- rejects unknown architecture keys and conflicting task metadata;
- loads SafeTensors on CPU or uses `torch.load(..., weights_only=True)` for the
  PyTorch fallback;
- uses strict state-dictionary loading with no key remapping;
- preserves the checkpoint floating dtype when no override is provided;
- honors explicit `dtype` and `device` conversion;
- leaves both loaded cores in evaluation mode; and
- has no network fallback or Hugging Face download call.

The public constructor now rejects:

- `pretrained=True` without an explicit local `checkpoint_path`;
- a checkpoint path when `pretrained=False`; and
- custom cores combined with `pretrained=True`.

This keeps license acceptance and weight acquisition outside the package. The
released TabFM weights remain covered by the TabFM Non-Commercial License
v1.0 and are not downloaded or redistributed by this implementation.

Added tests for:

- official classification/regression directory and configuration layout;
- SafeTensors loading;
- `pytorch_model.bin` fallback loading;
- exact loaded key and tensor equality for both variants;
- classification and regression inference equality after loading;
- an explicit monkeypatch that fails if local loading attempts a download;
- bfloat16 conversion and output dtype;
- missing regression variant rejection;
- task/directory mismatch rejection;
- unknown configuration key rejection;
- non-boolean `is_classifier` rejection; and
- strict missing-state-key rejection.

Synthetic tiny checkpoints were used for automated tests. They reproduce the
official directory, config, and state-dictionary format without copying or
redistributing licensed weights. A real official-weight smoke test remains
conditional on the user providing an already licensed local checkpoint.

Initial checkpoint/model command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q \
  test/models/tabfm/test_checkpoint.py test/models/tabfm/test_model.py
```

```text
30 passed in 2.38s
```

Focused TabFM command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q test/models/tabfm
```

```text
133 passed in 3.07s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q
```

```text
323 passed, 81 skipped in 8.54s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm sdm/models/__init__.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm sdm/models/__init__.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm sdm/models/__init__.py test/models/tabfm
git diff --check
```

Final result: all lint, format, type, and whitespace checks passed.

Issues found and resolved:

1. `safetensors` was not installed in the temporary test environment. It was
   added to `pyproject.toml` and version 0.8.0 was installed in
   `/tmp/tabfm-smoke-venv` for validation.
2. Initial Ruff validation identified the loader as an undocumented public
   function, one long exception line, and test import ordering. The loader was
   made private because it is an implementation detail of `TabFM`, the line was
   wrapped, and Ruff organized imports.
3. Ruff format requested routine formatting in `checkpoint.py`; formatting was
   applied and the subsequent format check passed.
4. `ty` inferred the synthetic checkpoint configuration too narrowly and
   reported two constructor diagnostics. Explicit `dict[str, Any]` typing
   resolved the inference issue; the final type check passed.

### 2026-07-08 — Tensor-native preprocessing recipes

Added and changed:

- [`sdm/processing/clamp.py`](sdm/processing/clamp.py)
- [`sdm/processing/__init__.py`](sdm/processing/__init__.py)
- [`sdm/models/tabfm/recipe.py`](sdm/models/tabfm/recipe.py)
- [`sdm/models/tabfm/model.py`](sdm/models/tabfm/model.py)
- [`sdm/models/tabfm/__init__.py`](sdm/models/tabfm/__init__.py)
- [`sdm/models/tabiclv2/recipe.py`](sdm/models/tabiclv2/recipe.py)
- [`test/processing/test_clamp.py`](test/processing/test_clamp.py)
- [`test/models/tabfm/test_recipe.py`](test/models/tabfm/test_recipe.py)

Added the generic stateless `Clamp` processor. It uses fixed constructor
bounds, requires no fitted state, and preserves tensor shape, dtype, and
device. This fills the shared gap between the existing fitted-quantile `Clip`
processor and the fixed `[-100, 100]` clipping required by the TabFM and
TabICLv2 standard-scaling paths.

Implemented public TabFM recipe factories:

```python
from sdm.models.tabfm import (
    default_classification_recipe,
    default_regression_recipe,
)
```

Both use the upstream single-estimator `"none"` normalization sequence for
numerical features:

```text
MeanImpute
  -> StandardScale(epsilon=1e-6)
  -> Clamp(-100, 100)
  -> SigmaClip(threshold=4)
```

The classification recipe keeps integer targets unchanged through `Identity`.
The regression recipe standard-scales context targets and supports inverse
transformation of predictions to the original target scale. `TabFM`'s
`default_recipe()` now returns this regression recipe instead of the temporary
identity placeholder.

Categorical indices and categorical masks intentionally bypass the numerical
feature pipeline. Callers fit and transform only the numerical `TableTensor`
block, then pass the untouched categorical block through the wrapper's existing
categorical routing. Learned feature and target statistics must be fitted on
context rows only.

The new generic `Clamp` was also inserted into the existing TabICLv2 regression
recipe at the previously documented upstream clipping point. This reuses one
shared processor rather than duplicating model-specific fixed-clipping code.

Added tests for:

- fixed two-sided, lower-only, and upper-only clamping;
- dtype and device preservation;
- invalid and reversed clamp bounds;
- exact TabFM numerical processor order and parameters;
- classification identity targets;
- regression target standardization and inverse round-trip;
- inverse transformation of new regression predictions;
- context-only feature state fitting;
- a control showing that including a query outlier would change learned state;
- finite context and outlier-query transforms;
- categorical indices remaining unchanged outside the numerical pipeline;
- `TabFM.default_recipe()` returning the regression recipe; and
- TabICLv2 behavior after adopting the shared clamp.

Initial targeted command and result:

```bash
/tmp/tabfm-smoke-venv/bin/pytest -q \
  test/processing/test_clamp.py \
  test/models/tabfm/test_recipe.py \
  test/models/test_tabiclv2.py
```

```text
18 passed, 6 skipped in 3.76s
```

Combined focused command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q \
  test/models/tabfm test/processing/test_clamp.py \
  test/models/test_tabiclv2.py
```

```text
151 passed, 6 skipped in 6.63s
```

Repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q
```

```text
334 passed, 81 skipped in 8.39s
```

The skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/processing sdm/models/tabfm sdm/models/tabiclv2/recipe.py \
  test/processing/test_clamp.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/processing sdm/models/tabfm sdm/models/tabiclv2/recipe.py \
  test/processing/test_clamp.py test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/processing sdm/models/tabfm sdm/models/tabiclv2/recipe.py \
  test/processing/test_clamp.py test/models/tabfm
git diff --check
```

Final result: all lint, format, type, and whitespace checks passed.

Issues found and resolved:

1. Initial Ruff validation found import ordering in both new test files and one
   long test name. Ruff organized the imports and the test was renamed without
   changing coverage.
2. Ruff format requested routine formatting in `test_recipe.py`; formatting
   was applied and the subsequent check passed.
3. `ty` could not infer that the generic `Recipe` fields were concrete
   `Sequential` processors in six test expressions. Narrow, explicit
   `cast(Sequential, ...)` bindings now document that tested runtime contract;
   the final type check passed.

### 2026-07-08 — Projected context cache

Changed:

- [`sdm/models/tabfm/attention.py`](sdm/models/tabfm/attention.py)
- [`sdm/models/tabfm/embedding.py`](sdm/models/tabfm/embedding.py)
- [`sdm/models/tabfm/icl.py`](sdm/models/tabfm/icl.py)
- [`sdm/models/tabfm/model.py`](sdm/models/tabfm/model.py)
- [`test/models/tabfm/test_cache.py`](test/models/tabfm/test_cache.py)

Extended TabFM attention with `KVCacheEntry` record/replay support without
changing parameter names or the uncached computation. `MultiheadAttention` can
now return normalized projected keys and projected values, then evaluate new
queries directly against those projections. `MultiheadAttentionBlock` and
`Encoder` propagate this lifecycle through residual and feed-forward blocks.

Column-stage caching records the projected context-derived inducing states used
by the second attention site of every induced block. On replay:

- the inducing-to-context attention is skipped;
- each query column attends directly to its cached inducing projections; and
- query states continue through every column block independently.

Both column stages have separate cache namespaces. Cache recording and replay
remain compatible with `col_chunk_size`: each folded column chunk records a
temporary cache, entries are concatenated across the folded table/column batch,
and replay slices the combined entries back into the same bounded chunks. This
preserves the released default column memory bound instead of materializing all
columns during prefill or decode.

Dataset-wise ICL caching records projected context keys and values for every
encoder block. The fitted context mask is cached separately, allowing padded
batches with different context lengths to replay without attending to invalid
context rows. Row interaction is intentionally not cached because each row is
processed independently and therefore has no reusable context attention state.

The public lifecycle is now:

```text
TabFM.fit(context, target)
  -> record both column-stage inducing projections
  -> record every ICL block's context K/V and context mask
  -> record schema/task/device/dtype metadata
  -> freeze cache

TabFM.predict(query)
  -> validate metadata
  -> execute query-only cell and row computation
  -> replay projected column and ICL context state
```

Raw context features and targets are no longer retained in the cache. Multiple
successive `predict` calls reuse the same frozen projections. The wrapper
rejects categorical-mask, active-width, batch-shape, feature-count, device, or
model-dtype changes that would invalidate replay.

Added tests for:

- attention projection record/replay in float32 and bfloat16;
- expected projected key/value tensor shapes;
- classification and regression prefill/decode parity;
- float32 and bfloat16 wrapper cache parity;
- categorical routing and padded active widths;
- repeated decode calls with different query row counts;
- the absence of raw `x` and `y` cache entries;
- frozen-cache mutation rejection;
- mixed per-table context lengths in direct core record/replay;
- replay masking of padded context rows;
- categorical-mask and active-width metadata mismatch rejection;
- model dtype changes after fitting; and
- rejection of encoder caching when RoPE is enabled, avoiding invalid replay
  position semantics; and
- chunked and unchunked column cache parity.

Initial cache test result:

```text
9 passed in 0.91s
```

After adding chunk-composable column entries and the RoPE guard, the final
cache result was:

```text
14 passed in 1.38s
```

The first complete validation before the chunk-composition refinement produced:

```text
148 TabFM tests passed in 3.47s
343 repository tests passed, 81 skipped in 8.40s
```

Final focused TabFM command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q test/models/tabfm
```

```text
153 passed
```

Final repository-wide command and result:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q
```

```text
348 passed, 81 skipped in 8.72s
```

The 81 skips remain the expected unavailable CUDA, CuPy, and cuDF cases. No
existing test regressed, and all uncached upstream parity tests still pass.

Static validation:

```bash
/tmp/tabfm-smoke-venv/bin/ruff check sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
git diff --check
```

Final result: all lint, format, type, and whitespace checks passed.

Issues found and resolved:

1. Initial static validation found two long wrapper lines, one test import-order
   issue, two files requiring formatting, and one optional attention-mask type
   narrowing issue in the uncached chunk path. Lines were wrapped, imports and
   formatting were normalized, and an explicit non-`None` assertion documents
   the branch invariant.
2. The first working cache path processed folded columns as one batch, which
   would bypass the released `col_chunk_size` memory bound for wide tables. The
   cache path was extended to record entries per chunk, concatenate projected
   state, and slice it during replay. Four additional task/dtype chunk tests
   passed, followed by the final focused and repository-wide runs.
3. Ruff format requested one final mechanical format update in the new chunk
   helper. It was applied and all subsequent static checks passed.

### 2026-07-08 — Compile, release-validation, and documentation coverage

Added and changed:

- [`test/models/tabfm/_compile_worker.py`](test/models/tabfm/_compile_worker.py)
- [`test/models/tabfm/test_compile.py`](test/models/tabfm/test_compile.py)
- [`test/models/tabfm/test_cuda.py`](test/models/tabfm/test_cuda.py)
- [`test/models/tabfm/test_official_checkpoint.py`](test/models/tabfm/test_official_checkpoint.py)
- [`docs/source/tabfm.md`](docs/source/tabfm.md)
- [`docs/source/index.md`](docs/source/index.md)

Compile coverage now runs representative `TabFMCore` paths through
`torch.compile(backend="eager", fullgraph=True, dynamic=True)`. It covers:

- classification and regression uncached forwards;
- chunked and unchunked column execution;
- projected-cache replay;
- a second row count through each compiled callable; and
- exact eager/compiled output comparisons.

`fullgraph=True` makes any graph break fail the test rather than silently
falling back to eager execution. Compile cases run in one short-lived worker
process so compiler allocations are released before the remainder of the
repository suite.

The official-checkpoint integration harness requires
`TABFM_CHECKPOINT_DIR`, skips with an explicit license/path reason when it is
unset, and never downloads weights. For both task variants it checks public
strict loading, deterministic inference, upstream output parity, cached replay,
and the absence of raw `x`/`y` cache entries.

The CUDA harness is guarded by the repository's `onlyCUDA` decorator. It
covers upstream and cache parity for classification and regression in float32,
bfloat16, and float16 where supported; chunked and unchunked column execution;
mixed context lengths and padded active widths; repeated query sizes; and peak
allocated-memory metadata for uncached, prefill, and replay execution. No CUDA
device was available in this environment, so the harness was collected and
skipped but not executed.

The new package guide documents the local checkpoint layout and license
boundary, classification and scalar-regression dispatch, `fit()` plus repeated
`predict()` use, numerical recipes, and cache schema/device/dtype/RoPE
constraints.

Tests:

```bash
/tmp/tabfm-smoke-venv/bin/pytest -o addopts='' \
  --import-mode=importlib -q test/models/tabfm/test_compile.py
```

```text
1 passed in 17.02s
```

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -o addopts='' \
  --import-mode=importlib -q test/models/tabfm
```

```text
154 passed, 16 skipped in 25.17s
```

The 16 focused skips are 14 CUDA cases and two official-checkpoint cases.

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -o addopts='' \
  --import-mode=importlib -q
```

```text
349 passed, 97 skipped in 25.81s
```

The existing CUDA, CuPy, and cuDF skips remain, with the new CUDA and licensed
checkpoint skips added explicitly.

Documentation and static validation:

```bash
/tmp/tabfm-smoke-venv/bin/sphinx-build -W -b html \
  docs/source /tmp/tabfm-docs-build
/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm
/tmp/tabfm-smoke-venv/bin/pip check
git diff --check
```

Final result: the warnings-as-errors documentation build, Ruff, format, type,
dependency, and whitespace checks all passed. `pre-commit run --all-files`
remains unavailable because its configured Python 3.10 interpreter is not
installed on this host.

Issues found and resolved:

1. The initial repository-wide command overrode `addopts` without restoring
   `--import-mode=importlib`. Pytest then reported three duplicate-basename
   collection errors. Restoring the repository's import mode resolved all
   three errors.
2. An initial eight-case compile matrix exhausted the constrained CPU runner
   after four separately retained compile graphs. Redundant combinations were
   removed while preserving both tasks, both column modes, cache replay, and
   dynamic row shapes.
3. Running compile graphs directly in the long-lived repository pytest process
   still caused memory pressure after later tests. The checks now run together
   in a subprocess worker; the final focused and repository-wide suites pass.
4. A multiprocessing-spawn attempt could not import the repository's `test`
   namespace because it conflicts with Python's standard-library `test`
   package. A direct worker script avoids that namespace ambiguity and fork
   warnings.

Next:

- execute the official-checkpoint harness after licensed local weights are
  supplied; and
- execute the CUDA parity/memory harness on a compatible GPU runner and record
  its device-specific results here.

### 2026-07-08 — Shared SDPA and rotary utilities

Changed:

- [`sdm/nn/attention.py`](sdm/nn/attention.py)
- [`sdm/nn/rope.py`](sdm/nn/rope.py)
- [`sdm/nn/__init__.py`](sdm/nn/__init__.py)
- [`sdm/models/tabfm/attention.py`](sdm/models/tabfm/attention.py)
- [`test/nn/test_attention.py`](test/nn/test_attention.py)
- [`test/nn/test_rope.py`](test/nn/test_rope.py)

The shared `SDPA` module now accepts an optional explicit logit `scale`. Its
default remains PyTorch's standard `1 / sqrt(head_channels)` behavior, while
TabFM configures `scale=1.0` after applying its checkpoint-compatible learned
per-dimension query scale. `SDPA` also accepts additive floating-point masks in
addition to boolean masks. QASSMax derives valid key counts from finite entries
when an additive mask is supplied.

Added the public `apply_rotary_embedding` tensor operation with `split_half`
and `interleaved` layouts. The existing shared `RotaryEmbedding` delegates to
the split-half layout, preserving TabICLv2 behavior. TabFM delegates to the
interleaved layout while retaining its model-specific `RotaryEmbedding`
wrapper and checkpoint-loaded `freqs` buffer.

TabFM's `MultiheadAttention` now delegates only the final attention kernel to
shared `SDPA`. It continues to own its separate q/k/v/out projections, q/k
RMSNorm modules, learned positive per-dimension scale, mask normalization,
and projected cache behavior. The shared module contains no parameters or
buffers in this configuration, so official state-dict names and keys are
unchanged.

Tests:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -o addopts='' \
  --import-mode=importlib -q \
  test/nn/test_rope.py test/nn/test_attention.py \
  test/models/tabfm/test_attention.py \
  test/models/tabfm/test_model.py test/models/tabfm/test_cache.py
```

```text
81 passed, 24 skipped in 2.70s
```

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -o addopts='' \
  --import-mode=importlib -q test/models/tabfm
```

```text
154 passed, 16 skipped in 23.16s
```

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -o addopts='' \
  --import-mode=importlib -q
```

```text
351 passed, 98 skipped in 27.93s
```

The warnings-as-errors Sphinx build, Ruff, format, type, dependency, and
whitespace checks also passed.

Issues found and resolved:

1. The first focused run had one shared-attention test that still expected a
   floating-point mask to be rejected. The test now verifies rejection of
   integer masks and separately verifies boolean/additive equivalence with
   QASSMax.
2. TabFM masks include an explicit singleton head dimension, whereas shared
   `SDPA` accepts masks shaped `[..., Q, KV]` and inserts that dimension.
   TabFM removes only that singleton dimension before delegation; upstream
   attention and end-to-end parity pass afterward.

Next:

- keep higher-level TabFM attention and transformer blocks model-specific;
  their normalization, projections, scaling, residual order, and feed-forward
  equations are not interchangeable with the current shared blocks.

### 2026-07-08 — Rebase onto current processor API

Changed:

- rebased the integration onto `origin/main` at `031319a`;
- resolved the TabICLv2 recipe overlap by retaining its new `StypeDispatch`
  and `ToNumerical` path and inserting the shared fixed-bound `Clamp` after
  standard scaling;
- updated `Clamp` to the current table-in/table-out `Processor` contract with
  numerical `Stype` validation and `_transform`; and
- updated Clamp and TabFM recipe tests to use numerical or categorical
  `TableTensor` inputs.

The first post-rebase focused run failed six recipe/processor cases because
the original Clamp implementation predated the new abstract `_transform`
contract. After adapting Clamp and the tests, the focused recipe checks passed:

```text
19 passed, 7 skipped in 3.99s
```

The full repository result on the rebased branch is:

```text
367 passed, 90 skipped in 29.47s
```

Next:

- execute the environment-dependent official-checkpoint and CUDA checks when
  their required assets are available.

## Current next step

Run the remaining release-readiness checks:

The ordered implementation checklist, commands, and acceptance criteria are in
[`TABFM_SDM_NEXT_STEPS.md`](TABFM_SDM_NEXT_STEPS.md).

Required work:

1. Run a local smoke/parity test with the actual official classification and
   regression SafeTensors after the user supplies an appropriately licensed
   checkpoint directory.
2. Run CUDA cache parity and peak-memory measurements when a CUDA runner is
   available.
3. Run `pre-commit run --all-files` in an environment with the configured
   Python 3.10 interpreter.

## Test history

| Date       | Scope                                     | Command                                                                                                                           | Result                                      |
| ---------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 2026-07-07 | Upstream classification smoke test        | `pytest test/models/tabfm/test_upstream_reference.py -vv`                                                                         | Passed                                      |
| 2026-07-07 | Upstream regression smoke test            | `pytest test/models/tabfm/test_upstream_reference.py -vv`                                                                         | Passed                                      |
| 2026-07-07 | New Python lint                           | `ruff check sdm/models/tabfm/__init__.py test/models/tabfm/test_upstream_reference.py`                                            | Passed                                      |
| 2026-07-07 | New Python format                         | `ruff format --check sdm/models/tabfm/__init__.py test/models/tabfm/test_upstream_reference.py`                                   | Passed                                      |
| 2026-07-07 | Whitespace                                | `git diff --check`                                                                                                                | Passed                                      |
| 2026-07-07 | Attention component parity                | `pytest test/models/tabfm -q`                                                                                                     | 15 passed                                   |
| 2026-07-07 | Full repository tests                     | `pytest`                                                                                                                          | 205 passed, 81 skipped                      |
| 2026-07-07 | TabFM Python lint                         | `ruff check sdm/models/tabfm/attention.py test/models/tabfm`                                                                      | Passed                                      |
| 2026-07-07 | TabFM Python format                       | `ruff format --check sdm/models/tabfm/attention.py test/models/tabfm`                                                             | Passed                                      |
| 2026-07-07 | TabFM type check                          | `ty check --python /tmp/tabfm-smoke-venv/bin/python sdm/models/tabfm test/models/tabfm`                                           | Passed                                      |
| 2026-07-07 | Pre-commit hooks                          | `pre-commit run --files ...`                                                                                                      | Not run: required Python 3.10 unavailable   |
| 2026-07-07 | Induced/column focused tests              | `pytest test/models/tabfm/test_embedding.py test/models/tabfm/test_attention.py test/models/tabfm/test_upstream_reference.py -vv` | 26 passed                                   |
| 2026-07-07 | Full suite after column embedding         | `pytest`                                                                                                                          | 216 passed, 81 skipped                      |
| 2026-07-07 | Column embedding lint/format/type         | Direct Ruff and `ty` commands                                                                                                     | Passed                                      |
| 2026-07-07 | MLP/cell focused tests                    | `pytest test/models/tabfm -q`                                                                                                     | 68 passed                                   |
| 2026-07-07 | Full suite after cell embedding           | `pytest`                                                                                                                          | 258 passed, 81 skipped                      |
| 2026-07-07 | MLP/cell lint/format/type                 | Direct Ruff and `ty` commands                                                                                                     | Passed                                      |
| 2026-07-07 | Row interaction focused tests             | `pytest test/models/tabfm -q`                                                                                                     | 90 passed                                   |
| 2026-07-07 | Full suite after row interaction          | `pytest`                                                                                                                          | 280 passed, 81 skipped                      |
| 2026-07-07 | Row interaction lint/format/type          | Direct Ruff and `ty` commands                                                                                                     | Passed                                      |
| 2026-07-07 | ICL focused tests                         | `pytest test/models/tabfm -q`                                                                                                     | 103 passed                                  |
| 2026-07-07 | Full suite after ICL                      | `pytest -q`                                                                                                                       | 293 passed, 81 skipped                      |
| 2026-07-07 | ICL lint/format/type                      | Direct Ruff and `ty` commands                                                                                                     | Passed                                      |
| 2026-07-07 | Core model parity                         | `pytest -q test/models/tabfm/test_model.py`                                                                                       | 11 passed                                   |
| 2026-07-07 | TabFM suite after core model              | `pytest -q test/models/tabfm`                                                                                                     | 114 passed                                  |
| 2026-07-07 | Full suite after core model               | `pytest -q`                                                                                                                       | 304 passed, 81 skipped                      |
| 2026-07-07 | Core lint/format/type/whitespace          | Direct Ruff, `ty`, and Git commands                                                                                               | Passed                                      |
| 2026-07-08 | Public wrapper tests                      | `pytest -q test/models/tabfm/test_model.py`                                                                                       | 22 passed                                   |
| 2026-07-08 | TabFM suite after public wrapper          | `pytest -q test/models/tabfm`                                                                                                     | 125 passed                                  |
| 2026-07-08 | Full suite after public wrapper           | `pytest -q`                                                                                                                       | 315 passed, 81 skipped                      |
| 2026-07-08 | Wrapper lint/format/type/whitespace       | Direct Ruff, `ty`, and Git commands                                                                                               | Passed                                      |
| 2026-07-08 | Local checkpoint/model tests              | `pytest -q test/models/tabfm/test_checkpoint.py test/models/tabfm/test_model.py`                                                  | 30 passed                                   |
| 2026-07-08 | TabFM suite after checkpoint loading      | `pytest -q test/models/tabfm`                                                                                                     | 133 passed                                  |
| 2026-07-08 | Full suite after checkpoint loading       | `pytest -q`                                                                                                                       | 323 passed, 81 skipped                      |
| 2026-07-08 | Checkpoint lint/format/type/whitespace    | Direct Ruff, `ty`, and Git commands                                                                                               | Passed                                      |
| 2026-07-08 | Clamp/recipe/TabICLv2 targeted tests      | Direct pytest command                                                                                                             | 18 passed, 6 skipped                        |
| 2026-07-08 | Combined preprocessing integration tests  | Direct pytest command                                                                                                             | 151 passed, 6 skipped                       |
| 2026-07-08 | Full suite after preprocessing recipes    | `pytest -q`                                                                                                                       | 334 passed, 81 skipped                      |
| 2026-07-08 | Preprocessing lint/format/type/whitespace | Direct Ruff, `ty`, and Git commands                                                                                               | Passed                                      |
| 2026-07-08 | Initial projected cache tests             | `pytest -q test/models/tabfm/test_cache.py`                                                                                       | 9 passed                                    |
| 2026-07-08 | Chunk-composable cache tests              | `pytest -q test/models/tabfm/test_cache.py`                                                                                       | 14 passed                                   |
| 2026-07-08 | TabFM suite after projected caching       | `pytest -q test/models/tabfm`                                                                                                     | 153 passed                                  |
| 2026-07-08 | Full suite after projected caching        | `pytest -q`                                                                                                                       | 348 passed, 81 skipped                      |
| 2026-07-08 | Cache lint/format/type/whitespace         | Direct Ruff, `ty`, and Git commands                                                                                               | Passed                                      |
| 2026-07-08 | Full-graph compile coverage               | `pytest -q test/models/tabfm/test_compile.py`                                                                                     | 1 passed                                    |
| 2026-07-08 | Official checkpoint harness               | `pytest -q test/models/tabfm/test_official_checkpoint.py`                                                                         | 2 skipped: no licensed checkpoint directory |
| 2026-07-08 | CUDA parity/memory harness                | `pytest -q test/models/tabfm/test_cuda.py`                                                                                        | 14 skipped: CUDA unavailable                |
| 2026-07-08 | TabFM suite after release coverage        | `pytest -q test/models/tabfm`                                                                                                     | 154 passed, 16 skipped                      |
| 2026-07-08 | Full suite after release coverage         | `pytest -q`                                                                                                                       | 349 passed, 97 skipped                      |
| 2026-07-08 | Documentation build                       | `sphinx-build -W -b html docs/source /tmp/tabfm-docs-build`                                                                       | Passed                                      |
| 2026-07-08 | Final static/dependency/whitespace checks | Direct Ruff, format, `ty`, `pip check`, and Git commands                                                                          | Passed                                      |
| 2026-07-08 | Shared SDPA/RoPE focused parity           | Focused shared NN, upstream, model, and cache tests                                                                               | 81 passed, 24 skipped                       |
| 2026-07-08 | TabFM suite after shared NN refactor      | `pytest -q test/models/tabfm`                                                                                                     | 154 passed, 16 skipped                      |
| 2026-07-08 | Full suite after shared NN refactor       | `pytest -q`                                                                                                                       | 351 passed, 98 skipped                      |
| 2026-07-08 | Shared NN refactor static/docs checks     | Sphinx, Ruff, format, `ty`, `pip check`, and Git commands                                                                         | Passed                                      |
| 2026-07-08 | Rebased processor integration             | Focused TabFM recipe, Clamp, and TabICLv2 tests                                                                                   | 19 passed, 7 skipped                        |
| 2026-07-08 | Full suite after rebase                   | `pytest -q`                                                                                                                       | 367 passed, 90 skipped                      |
| 2026-07-08 | Rebased static/docs checks                | Sphinx, Ruff, format, `ty`, Markdown format, `pip check`, and Git commands                                                        | Passed                                      |

## Known limitations and open decisions

- Attention, MLP, cell embedding, induced set attention, column embedding, row
  interaction, ICL, and the checkpoint-compatible neural core have been
  implemented. The public SDM wrapper and exports are also implemented.
  Strict local checkpoint loading and the minimum numeric preprocessing path
  are implemented. Projected context caching is implemented for both column
  stages and every ICL layer. Advanced estimator preprocessing remains.
- End-to-end core parity now passes against the pinned upstream implementation.
  The public wrapper returns query rows and preserves TabFM-specific metadata.
- The temporary test environment is not part of the repository and may need to
  be recreated in a later session.
- Official checkpoint download behavior remains blocked on maintainer approval
  of the non-commercial weight-license flow. Explicit local loading is
  available.
- Real official weights were not present in the workspace, so checkpoint tests
  use synthetic tiny models with the exact official format. Run a real-weight
  smoke test when an appropriately licensed local checkpoint is supplied.
- The wrapper extends `BaseModel.forward`, `fit`, and `predict` with optional
  TabFM metadata rather than changing the shared base API.
- The wrapper now exposes the upstream single-estimator `"none"` numerical
  preprocessing path. Power, quantile, robust, ensemble permutation,
  augmentation, and calibration variants from the full estimator remain out of
  scope for this lightweight core integration.
- Recipe application is explicit: the current `BaseModel` boundary does not
  automatically fit and apply a returned recipe. Callers must fit learned
  processors on context rows and keep categorical indices outside the numeric
  pipeline.
- Cached column execution preserves memory bounds through chunk-composable
  entries, but CUDA peak-memory and throughput measurements have not yet been
  run in this CPU-only environment.
- Full-graph compile coverage passes with the eager backend. Inductor remains
  an optional environment-specific smoke test.
- TabFM now reuses shared SDPA and rotary tensor operations without changing
  checkpoint keys or substituting TabICLv2 transformer equations.
- Official-checkpoint and CUDA validation harnesses are present, but their
  environment-dependent executions remain pending.
- The first implementation will preserve TabFM's architecture. QASSMax, direct
  grouped projection, one column/row cycle, and quantile regression are later
  model variants rather than checkpoint-compatible refactors.

## Log-entry template for future work

Copy this section for every meaningful integration update:

```markdown
### YYYY-MM-DD — Short change title

Changed:

- files and behavior changed;
- architectural or API decisions; and
- any license/dependency implications.

Tests:

- exact command;
- pass/fail/skip counts;
- relevant dtype/device/backend; and
- parity tolerance when applicable.

Issues found:

- failures or unexpected behavior and their cause;
- corrective changes; and
- unresolved risks.

Next:

- one concrete next implementation step.
```

Never replace a failed result with only the later successful result. Retain the
failure and resolution when it reveals an integration constraint or regression
risk.
