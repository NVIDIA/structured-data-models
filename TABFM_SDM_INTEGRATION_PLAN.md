# TabFM integration plan for SDM

## Objective

Add a native, PyTorch-first TabFM implementation to the `structured-data-models`
package while preserving the repository's tensor-centric, modular, and
lightweight design.

The first implementation must reproduce the released TabFM PyTorch model closely
enough to load its checkpoints and pass output-parity tests. TabICLv2-inspired
architecture changes should be introduced only later as separately configured
and trained model variants.

Read [`TABICLV2_TABFM_COMPARISON.md`](TABICLV2_TABFM_COMPARISON.md) before
starting. It documents the shared architecture, meaningful differences, and
which substitutions are checkpoint-safe.

## Source and licensing constraints

Use these upstream sources:

- TabFM source commit:
  [`633cd265f498e1d20c9625be0639f6305d8e2541`](https://github.com/google-research/tabfm/tree/633cd265f498e1d20c9625be0639f6305d8e2541)
- PyTorch checkpoint repository:
  [`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch)

The source repository is Apache 2.0. Preserve applicable Google copyright and
Apache license headers in adapted files, retain attribution, and mark adapted
files as modified.

The released weights are governed by the **TabFM Non-Commercial License v1.0**,
not Apache 2.0. Do not redistribute the weights as part of the SDM package. Do
not enable unconditional or implicit checkpoint download until the maintainers
have approved how the non-commercial license will be surfaced and accepted.
Keep `pretrained=False` usable without downloading any restricted asset.

## Scope of the first pull request

Include:

- a native PyTorch TabFM neural model;
- classification and scalar-regression variants;
- official checkpoint loading or a checkpoint remapping utility, subject to the
  licensing gate above;
- direct tensor inference with a context/test row split;
- unit tests for shapes, masking, information flow, and padding;
- parity tests against the pinned upstream PyTorch implementation; and
- public exports from `sdm.models`.

Do not include in the first pull request:

- JAX/Flax or Orbax support;
- sklearn-compatible estimators;
- pandas, scipy, or scikit-learn as mandatory dependencies;
- OOF prediction, NNLS weighting, SVD features, feature crosses, or probability
  calibration;
- automatic checkpoint redistribution;
- TabICLv2 substitutions that alter TabFM's learned function; or
- platform, serving, or deployment abstractions.

## Development setup

Clone TabFM as a sibling or temporary reference checkout. Do not nest its Git
repository inside this repository and do not add it as a submodule.

```bash
cd /home/ruthvikak
git clone https://github.com/google-research/tabfm.git tabfm-reference
git -C tabfm-reference checkout 633cd265f498e1d20c9625be0639f6305d8e2541
git -C tabfm-reference rev-parse HEAD

cd /home/ruthvikak/structured-data-models
git switch -c add-tabfm
```

The reference clone is used only to inspect source and generate parity outputs.
It must not become a package dependency.

## Target package structure

Start with the following structure, adjusting boundaries only when a different
split materially improves reuse:

```text
sdm/models/tabfm/
├── __init__.py
├── attention.py
├── embedding.py
├── icl.py
├── model.py
└── recipe.py

test/models/tabfm/
├── test_attention.py
├── test_embedding.py
├── test_model.py
└── test_parity.py
```

Keep TabFM-specific equations under `sdm/models/tabfm` initially. Move a block
to `sdm.nn` only if it is generic, has a stable typed interface, and can be
shared without changing TabFM checkpoint behavior.

In `__init__.py`, order imports and `__all__` by dependency order rather than
alphabetically.

## Implementation sequence

### Step 1: Record the upstream contract

Before editing model code, record the pinned TabFM v1.0.0 configuration:

| Parameter               |   Value |
| ----------------------- | ------: |
| Maximum classes         |      10 |
| Cell embedding width    |     256 |
| Feature group size      |       3 |
| Fourier frequencies     |      32 |
| Column blocks per stage |       3 |
| Column attention heads  |       4 |
| Inducing points         |     256 |
| Row blocks per stage    |       3 |
| Row attention heads     |       8 |
| Readout/CLS tokens      |       8 |
| ICL blocks              |      24 |
| ICL heads               |       8 |
| Feed-forward factor     |       4 |
| Activation              |  SwiGLU |
| RoPE base               | 100,000 |

Document the core tensor contract:

```text
x:          [batch, total_rows, features]
y:          [batch, total_rows] upstream, with only context values consumed
train_size: [batch]
cat_mask:   optional [batch, features]
d:          optional [batch] active feature counts before padding

classification output: [batch, total_rows, 10]
regression output:     [batch, total_rows, 1]
```

Decide whether the public SDM wrapper will return all rows or only test rows.
The preferred SDM contract is test rows only, matching `BaseModel`. Keep the
internal core capable of producing the upstream full sequence so parity tests
remain straightforward.

### Step 2: Add TabFM-specific transformer primitives

Implement the exact PyTorch operations used by the released model:

- RMSNorm with the same epsilon and float32 accumulation behavior;
- rotary embeddings using a checkpoint-loadable frequency buffer;
- separate q, k, v, and output projections;
- per-head q/k RMSNorm;
- learned positive per-dimension query scaling;
- scaled dot-product attention with scale `1.0` after explicit query scaling;
- residual multi-head attention blocks;
- SwiGLU feed-forward blocks; and
- optional exact activation chunking for memory control.

Do not replace these with TabICLv2's `TransformerBlock`, LayerNorm, GELU, fused
qkv projection, or QASSMax in the checkpoint-compatible implementation.

Tests must cover:

- output shapes and dtypes;
- CPU float32 forward execution;
- bfloat16 execution where supported;
- attention masks;
- RoPE position behavior; and
- parity with the matching upstream primitive using identical parameters.

### Step 3: Implement induced column attention

Implement:

```text
inducing_state = attention(inducing_tokens, context_rows)
row_state = attention(input_rows, inducing_state)
```

Stack the induced blocks into a set transformer and wrap them in a column
embedding stage that:

1. transposes `[B, T, H, E]` to treat `B * H` columns as the batch axis;
2. masks keys after each member's `train_size`;
3. applies the set-transformer stack;
4. applies TabFM's output projection and RMSNorm; and
5. restores `[B, T, H, E]`.

Preserve the option to chunk the independent column axis without changing
outputs.

Tests must verify that modifying test rows does not alter the context-derived
inducing summary used for other test rows.

### Step 4: Implement the cell embedder

Preserve the released TabFM behavior:

- group offsets are `(2**i) - 1`, producing `[0, 1, 3]` for group size 3;
- indices wrap by active feature count `d` when supplied;
- numerical and categorical values use separate learned Fourier-frequency
  banks and projections;
- sine and cosine are computed in float32;
- grouped slot embeddings are summed;
- padded feature columns are zeroed after grouping;
- classification targets use a class embedding;
- regression targets use the released target MLP; and
- target embeddings are added only to rows before `train_size`.

Do not use TabICLv2's `[1, 2, 4]` offsets or direct grouped linear projection in
this checkpoint-compatible path.

Tests must cover numerical-only data, mixed categorical/numerical masks,
different active widths `d`, padded features, and absence of test-label
information flow.

### Step 5: Implement row interaction

Implement row-wise attention by flattening batch and row axes and attending
over feature tokens.

Preserve:

- learnable CLS/readout tokens;
- RoPE base 100,000;
- masks for padded features using `d + num_cls`;
- a first row stage that returns the full sequence; and
- a second row stage that returns and flattens only CLS tokens.

The two row stages have the same block design but different output behavior.
Do not collapse them into one TabICLv2-style row embedding in the first model.

After parity is established, final-layer CLS-only query pruning may be added as
an optimization with a strict output-parity test.

### Step 6: Implement dataset-wise ICL and task heads

The ICL module must:

1. encode context targets;
2. add target embeddings only before each member's `train_size`;
3. restrict attention keys/values to context rows;
4. omit positional embeddings across rows;
5. apply final RMSNorm; and
6. decode classification to 10 logits or regression to one scalar.

Preserve TabFM's target encoders for initial checkpoint parity. A classification
`OneHotAndLinear` encoder may later be converted exactly to an embedding using:

```text
embedding.weight = projection.weight.T + projection.bias
```

This conversion is exact only for valid class IDs unless an explicit unknown
embedding is also retained.

### Step 7: Assemble the core model

Assemble the stages in this exact order:

```text
cell_embedder
  -> col_embedder
  -> prepend CLS tokens
  -> row_interactor (full sequence)
  -> col_embedder_2
  -> row_interactor_2 (CLS outputs only)
  -> icl_predictor
```

At model entry:

- replace NaNs with the upstream `-100` sentinel;
- cast to model compute dtype;
- retain `cat_mask` and `d`; and
- validate class count and tensor shapes.

Keep architecture defaults explicit and typed. Avoid introducing a mandatory
configuration-object API; users should be able to compose the Python modules
directly.

### Step 8: Add the SDM public wrapper

Expose a public `TabFM` model from:

```text
sdm/models/tabfm/__init__.py
sdm/models/__init__.py
```

Use `BaseModel` only if its current `forward`/`fit`/`predict` interface can
preserve TabFM's `cat_mask`, `d`, and per-batch train sizes without hiding or
discarding information. Otherwise, make the minimal generic extension to
`BaseModel` or introduce an internal core plus a thin TabFM adapter.

Do not put sklearn estimator behavior in the model wrapper. The wrapper should:

- select classification or regression by target dtype or an explicit task;
- return test-row predictions;
- expose `pretrained` and `device` consistently with `TabICLv2`;
- remain usable with `pretrained=False`; and
- keep model-family-specific logic thin.

If `TableTensor` inputs are supported, categorical columns must not be silently
dropped. Preserve a categorical mask through preprocessing and define how
categorical values become tensor IDs.

### Step 9: Add checkpoint loading

Prefer direct loading from the official PyTorch checkpoint when parameter names
and shapes can be preserved. Otherwise, add a deterministic remapper similar to
TabICLv2's `_remap_ckpt`.

Checkpoint tests must verify:

- no missing or unexpected keys after approved remapping;
- classification and regression variants load separately;
- loaded model dtype and device are correct;
- offline/local-cache loading works; and
- `pretrained=False` performs no network access.

Because the weights are non-commercial, put automatic remote download behind a
maintainer-approved policy. At minimum, document the weight license adjacent to
the loader and in the model documentation.

### Step 10: Add preprocessing without heavy core dependencies

Implement the minimum tensor-native recipe needed for numeric input:

```text
features:
  MeanImpute
  -> StandardScale(epsilon=1e-6)
  -> explicit Clip(-100, 100), if exact TabFM behavior is required
  -> SigmaClip(threshold=4)

regression target:
  StandardScale with inverse transform
```

Do not add pandas, scipy, or scikit-learn to core dependencies merely to copy
TabFM's estimator pipeline. Mixed-type dataframe ergonomics should be expressed
through `TableTensor` and composable SDM processors where possible.

Treat these as later optional processors or packages:

- categorical ordinal encoding;
- datetime expansion;
- constant-feature filtering;
- power/quantile normalization choices;
- feature and class permutations;
- row/feature subsampling;
- SVD and feature-cross augmentation;
- OOF prediction, NNLS, and calibration.

Any fitted preprocessing state must be learned from context/training rows only.

### Step 11: Establish upstream parity

Create a parity test that instantiates the pinned upstream PyTorch model and the
SDM model with the same small configuration, copies or remaps every parameter,
and compares outputs.

Cover at least:

- classification and regression;
- numerical-only input;
- categorical masks;
- padded feature widths with `d`;
- different train sizes across batch members;
- float32 CPU inference; and
- bfloat16 inference on supported hardware.

Use tolerances appropriate to dtype. Float32 parity should be tight enough to
detect block-order, normalization, RoPE, mask, and projection errors. Keep the
upstream checkout outside the installed package; parity tests may be marked as
optional/integration tests if upstream dependencies are not available in normal
CI.

### Step 12: Add caching after forward parity

Do not implement caching until the uncached forward path passes parity.

Add a record/freeze/replay lifecycle for:

- first column-stage inducing representations or projected K/V;
- second column-stage inducing representations or projected K/V; and
- every ICL layer's context K/V.

Cache metadata must include or validate:

- context row counts;
- active feature counts;
- categorical masks or feature schema;
- dtype and device assumptions; and
- batch dimensions.

Required test:

```text
uncached(context + test) == decode(test, prefill(context))
```

Run this test for both tasks, mixed train sizes, padded feature widths, and
multiple successive decode calls.

## Required test matrix

| Area             | Required checks                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------- |
| Construction     | Defaults, custom small dimensions, CPU/CUDA device placement                                                |
| Cell embedding   | Group offsets, Fourier math, categorical routing, `d` masking                                               |
| Column attention | Context-only keys, induced block shapes, chunk parity                                                       |
| Row attention    | CLS placement, feature padding mask, RoPE behavior                                                          |
| ICL              | Context-label injection, context-only keys, no test-label leakage                                           |
| Outputs          | Classification `[... , test_rows, 10]`, regression `[... , test_rows, 1]` or squeezed documented equivalent |
| Checkpoints      | Complete load for both official variants, license-safe failure messages                                     |
| Caching          | Prefill/decode parity and repeated decode correctness                                                       |
| Upstream parity  | Small and default-like configurations for both tasks                                                        |
| Processing       | Train-only fitted state and regression inverse transform                                                    |
| Compilation      | Exercise a representative forward under `torch.compile` if practical                                        |

Run focused tests during development, followed by repository-wide checks:

```bash
pytest test/models/tabfm -vv
pytest
pre-commit run --all-files
```

## Coding requirements

- Type all public function and method boundaries.
- Preserve tensor device and dtype; avoid `.cpu()`, `.numpy()`, `.item()`, and
  newly created CPU tensors in model execution.
- Prefer tensor methods and vectorized PyTorch operations.
- Use keyword arguments in multiline calls.
- Avoid `else` after `return`, `raise`, `break`, or `continue`.
- Add short shape comments for non-obvious transposes and reshapes.
- Keep direct Python composition as the primary interface.
- Do not make stochastic processing implicit; expose a seed or generator.
- Document all public constructor parameters.
- Keep model wrappers thin and put genuinely shared abstractions outside the
  model package only after their reuse is demonstrated.

## Suggested commit sequence

Keep reviewable commits aligned with independently testable milestones:

1. `Add TabFM attention primitives`
2. `Add TabFM cell and column embeddings`
3. `Add TabFM row interaction and ICL blocks`
4. `Add TabFM model and task heads`
5. `Add TabFM checkpoint conversion and parity tests`
6. `Add TabFM SDM wrapper and preprocessing recipe`
7. `Add TabFM context caching`

Do not mention tool attribution in commit messages or PR metadata.

## Completion criteria

The first integration is complete when:

- `from sdm.models import TabFM` works;
- `TabFM(pretrained=False)` runs classification and regression on tensors;
- official classification and regression checkpoints can be loaded under the
  approved license flow;
- upstream-to-SDM float32 parity passes for representative inputs;
- categorical masks, active feature widths, and train/test boundaries are
  preserved;
- context/test leakage tests pass;
- caching, if included in the first PR, matches uncached inference;
- no JAX, pandas, scipy, or scikit-learn dependency is added to the core package;
- all new public APIs are documented and typed; and
- `pytest` and `pre-commit run --all-files` pass.

## Follow-up work after the faithful integration

Only after checkpoint-compatible parity is stable, consider separately tested
TabICLv2-inspired variants:

- final-layer CLS-only and test-row-only query pruning;
- projected inducing K/V caching;
- direct grouped linear cell embedding;
- TabICLv2's `[1, 2, 4]` grouping schedule;
- QASSMax attention scaling;
- one column/row cycle instead of two;
- LayerNorm/GELU blocks;
- smaller readout representations; and
- quantile regression.

Except for query pruning and cache representation changes proven by parity,
these variants change the learned function and require new training or
fine-tuning. Do not load released TabFM v1.0.0 weights into them and claim model
equivalence.
