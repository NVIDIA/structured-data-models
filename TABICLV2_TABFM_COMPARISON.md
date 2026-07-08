# TabICLv2 and TabFM: structural comparison and substitution guide

## Scope and source snapshot

This report compares:

- the TabICLv2 implementation in this repository at commit
  `961f5f80723fdbe7e5727b3a88c3d31b9c1f9687`, primarily
  [`sdm/models/tabiclv2`](sdm/models/tabiclv2), the shared
  [`sdm/nn`](sdm/nn) blocks, [`sdm/cache.py`](sdm/cache.py), and
  [`sdm/processing`](sdm/processing); and
- Google Research TabFM at commit
  [`633cd265f498e1d20c9625be0639f6305d8e2541`](https://github.com/google-research/tabfm/tree/633cd265f498e1d20c9625be0639f6305d8e2541),
  including its JAX model, PyTorch model, estimator wrappers, preprocessing,
  ensembling, and prefill/decode paths.

“Similar” below means that two sections play the same architectural or runtime
role. It does **not** imply identical equations, tensor values, parameters, or
checkpoint compatibility. The final substitution section separates safe
implementation changes from architecture changes that require retraining.

## Executive summary

TabICLv2 and TabFM share the same broad in-context tabular processing pattern:

```text
raw table
  -> overlapping shifted feature groups
  -> per-cell/group embedding + training-label injection
  -> column-wise induced set attention across rows
  -> learnable readout/CLS tokens + row-wise attention across features
  -> fixed-width row representation
  -> label-conditioned dataset-wise attention across rows
  -> task head
```

TabFM extends this pattern in four important ways:

1. Its current cell embedder uses learned Fourier features and separate
   numerical/categorical banks rather than TabICLv2's direct grouped linear
   projection.
2. It performs **two** alternating column/row cycles. TabICLv2 performs one
   column stage followed by one row stage.
3. Its transformer math uses RMSNorm, per-dimension query scaling, q/k
   normalization, and SwiGLU. TabICLv2 uses LayerNorm, GELU, and QASSMax in the
   column and ICL stages.
4. Its regression model emits one scalar. TabICLv2 emits 999 quantiles.

Consequently, the two implementations are structurally close, but current
TabFM v1.0.0 weights are not drop-in weights for TabICLv2 modules.

## End-to-end stage map

| Role                                  | TabFM symbol/path                                                                                                                                                | TabICLv2 symbol/path                                                                               | Similarity                                                                                                                   | Material difference                                                                                                              |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Top-level task model                  | [`TabFM.forward`](https://github.com/google-research/tabfm/blob/633cd265f498e1d20c9625be0639f6305d8e2541/tabfm/src/pytorch/model.py#L473) / JAX `TabFM.__call__` | [`_TabICLv2.forward`](sdm/models/tabiclv2/model.py#L207)                                           | Same staged ICL pipeline                                                                                                     | TabFM has two column/row cycles and passes `cat_mask`, active feature count `d`, and per-batch `train_size`                      |
| Cell/group embedding                  | TabFM `CellEmbedder`                                                                                                                                             | TabICLv2 `RowEmbedding` lines 91–105                                                               | Groups neighboring columns and injects labels only into context rows                                                         | TabFM uses Fourier/type-specific embedding; TabICLv2 uses one `Linear(G, D)`                                                     |
| Feature grouping                      | TabFM `CellEmbedder._group` / JAX `feature_grouping`                                                                                                             | Inline index construction in `RowEmbedding.forward`                                                | Cyclic, exponentially spaced, overlapping feature groups                                                                     | TabFM offsets are `2**i - 1` = `[0,1,3,…]`; TabICLv2 offsets are `2**i` = `[1,2,4,…]`                                            |
| Context-label embedding at cell stage | TabFM `y_embedder_lookup`                                                                                                                                        | `RowEmbedding.y_emb` / `y_lin`                                                                     | Classification lookup and continuous regression projection are added to every feature token of training rows                 | TabFM regression uses a one-hidden-layer MLP; TabICLv2 uses a linear layer                                                       |
| Column-wise distribution encoder      | TabFM `ColEmbedding` + `SetTransformer`                                                                                                                          | `RowEmbedding.col_layers`                                                                          | Each feature is a batch item; rows are the set axis; context rows determine the representation                               | TabFM adds output projection + RMSNorm; TabICLv2 uses QASSMax and LayerNorm inside its blocks                                    |
| Induced set attention                 | TabFM `InducedSelfAttentionBlock`                                                                                                                                | [`InducedTransformerBlock`](sdm/nn/set_transformer.py#L15)                                         | Same two-step Set Transformer equation: inducing tokens attend to rows, then rows attend to inducing tokens                  | Underlying normalization, activation, and query scaling differ                                                                   |
| Readout tokens                        | TabFM `cls_tokens`                                                                                                                                               | `RowEmbedding.readout_token`                                                                       | Learnable tokens aggregate a variable-width row into a fixed-width vector                                                    | TabFM v1.0.0 uses 8; TabICLv2 uses 4                                                                                             |
| Row-wise feature interaction          | TabFM `RowInteraction` + `Encoder`                                                                                                                               | `RowEmbedding.row_layers`                                                                          | Flattens batch × rows and attends over feature/readout tokens with RoPE                                                      | TabFM's first row stage returns the full token sequence for another cycle; TabICLv2's last row layer queries only readout tokens |
| Second column/row cycle               | `col_embedder_2`, `row_interactor_2`                                                                                                                             | No counterpart                                                                                     | Repeats the same kinds of operations                                                                                         | This is TabFM-only and cannot be deleted while retaining its checkpoint behavior                                                 |
| Dataset-wise ICL                      | TabFM `ICLearning`                                                                                                                                               | [`ICLBlock`](sdm/models/tabiclv2/icl.py#L13)                                                       | Adds encoded labels to context-row representations, restricts keys to context rows, and attends across examples without RoPE | TabFM computes all query positions in every layer; TabICLv2 queries only test rows in the last layer                             |
| Prediction decoder                    | `ICLearning.decoder`                                                                                                                                             | `_TabICLv2.head`                                                                                   | Two-layer MLP after final normalization                                                                                      | Classification width agrees at 10; regression width is 1 in TabFM and 999 in TabICLv2                                            |
| Context reuse                         | JAX `TabFM.prefill` / `decode` and `ICLearningCache`                                                                                                             | [`BaseModel.fit` / `predict`](sdm/models/base.py#L50), [`Cache`](sdm/cache.py#L49), `KVCacheEntry` | Separates context preprocessing from query decoding and reuses context-derived attention state                               | Current TabFM PyTorch `TabFM` has no equivalent public prefill/decode path; JAX cache structure is model-specific                |

## Function- and strategy-level similarities

### 1. Feature grouping and cell embedding

Both models deliberately break pure feature-permutation symmetry by creating an
overlapping group for every original feature position and wrapping indices at
the feature boundary. Both use exponentially spaced offsets, so distant
features can enter a group without increasing group size.

The implementations are not index-identical:

```text
TabFM, group size 3:    base feature + offsets [0, 1, 3]
TabICLv2, group size 3: base feature + offsets [1, 2, 4]
```

TabFM also wraps by the active feature count `d` for padded heterogeneous
batches. TabICLv2 wraps by the physical column count and currently assumes all
batch members have the same active width.

After grouping, both models form one embedding per row and feature group. The
current difference is the embedding function:

- TabFM `CellEmbedder._cell` computes learned sine/cosine Fourier features,
  uses separate numerical and categorical frequency/projection banks, and sums
  the group slots.
- TabICLv2 directly projects the vector of grouped raw values with
  `Linear(group_size, channels)`.

TabFM's JAX `CellEmbedder` still contains a `use_fourier_features=False` legacy
path that performs a direct linear projection. This is the closest TabFM code
path to TabICLv2's cell embedding, although the group offsets still differ.

### 2. Early target injection

Both models inject the target twice:

1. into each training-row cell/group embedding before column-wise attention;
2. into each fixed-width training-row representation before dataset-wise ICL.

In both places, test rows never receive their own target embeddings. This is a
substantive common strategy: column statistics become target-aware early, and
the final dataset transformer is explicitly label-conditioned.

The classification encoders are algebraically related. TabFM's
`OneHotAndLinear(K, D)` returns `one_hot(y) @ W^T + b`; an `Embedding(K, D)`
with weight `W^T + b` returns the same value for valid class IDs. TabICLv2's
checkpoint remapper already uses exactly this conversion for its source
checkpoint in [`_remap_ckpt`](sdm/models/tabiclv2/model.py#L256). The same
conversion can be used when replacing TabFM's **ICL classification** target
encoder with an embedding. It does not directly cover TabFM's cell-stage
lookup, which is already an embedding.

### 3. Column-wise induced set attention

The common algorithm is:

```text
H = attention(inducing_tokens, context_rows)
output_rows = attention(input_rows, H)
```

This changes quadratic attention over rows into work proportional to
`(rows + context_rows) * inducing_points`. Both stack several such blocks and
share the block weights across columns. Both also support using only training
rows when producing the inducing representation, preventing query/test rows
from changing the learned context summary.

The cache representations are mathematically related:

- TabFM JAX caches each layer's hidden inducing representation.
- TabICLv2 caches the projected keys and values of that hidden representation
  for the second attention half.

The latter skips the repeated key/value projection as well as the first
attention half during decode.

### 4. Row-wise attention and readout tokens

Both prepend learnable tokens to the per-feature sequence, apply self-attention
within each row, then flatten the selected readout tokens to create a fixed
row width:

```text
row representation width = number of readout tokens * cell embedding width
```

Both use rotary position embeddings with base/theta 100,000 in this stage and
do not use RoPE in the column-wise or dataset-wise stages. Thus, column order is
made visible to row attention while the row set remains position-free for ICL.

TabICLv2 avoids unnecessary output computation by making only the readout
tokens the queries in the final row layer. TabFM's second `RowInteraction`
currently computes the full sequence and slices its CLS tokens afterwards.
This optimization is a strong candidate for reuse, subject to an output-parity
test.

### 5. Dataset-wise in-context learning

The common ICL operation is:

```text
context_state = row_representation + encoded_target
test_state = row_representation
keys/values = context states only
queries = context and test states (or only test states in the final layer)
```

Both use a transformer stack without positional encoding, so the context rows
are treated as a set. Both can reuse per-layer context keys and values. Both
normalize the final states and decode them with a small MLP.

TabICLv2 uses QASSMax for all ICL layers. TabFM instead uses q/k RMSNorm plus a
learned, positive per-head-dimension query scale. These mechanisms occupy the
same attention-temperature design space, but they are not equivalent:

- TabFM's scale does not depend on context length or the current query value.
- TabICLv2's QASSMax depends on `log(number_of_keys)` and includes a
  query-dependent gate.

### 6. Attention and transformer primitives

| TabFM                       | TabICLv2/shared SDM                         | Shared role                                                | Non-equivalence to preserve in a port                                                                   |
| --------------------------- | ------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `MultiheadAttention`        | `Attention` + `SDPA`                        | q/k/v projections, multi-head SDPA, output projection      | TabFM has separate projections, q/k RMSNorm, and `PerDimScale`; SDM uses fused qkv and optional QASSMax |
| `MultiheadAttentionBlock`   | `TransformerBlock`                          | residual attention followed by residual FFN                | TabFM v1 uses RMSNorm and SwiGLU with four norm sites; TabICLv2 uses LayerNorm and GELU pre-norm        |
| `InducedSelfAttentionBlock` | `InducedTransformerBlock`                   | two cross-attention blocks through learned inducing points | Same topology, different block math                                                                     |
| `Encoder`                   | `ModuleList[TransformerBlock]` loops        | stack transformer blocks                                   | TabFM owns shared RoPE at encoder level; TabICLv2 passes one `RotaryEmbedding` through its row loop     |
| `RoPE` / `rope_interleaved` | `RotaryEmbedding`                           | rotate projected q/k by feature position                   | Frequency-buffer layout and checkpoint naming differ                                                    |
| `RMSNorm`                   | `LayerNorm`                                 | stabilize residual blocks and final state                  | Different normalization equation and parameter set                                                      |
| `MLP`                       | `torch.nn.Sequential(Linear, GELU, Linear)` | target encoders and decoders                               | TabFM transformer FFNs use SwiGLU; its standalone decoder MLP uses GELU                                 |

### 7. Task dispatch and output contracts

Both instantiate separate pretrained classification and regression variants,
encode classification labels, standardize regression targets, and apply a
small task decoder. The output contracts differ:

- classification: both support up to 10 logits in their released/default
  configurations;
- regression: TabFM returns one standardized point estimate, whereas TabICLv2
  returns 999 values interpreted as quantiles at levels 0.001 through 0.999.

Changing TabFM to the TabICLv2 regression head therefore also requires changing
its estimator's output-dimension check, inverse target transform, ensemble
combination, OOF/NNLS logic, and public `predict` policy (for example, selecting
the median quantile for a point prediction).

### 8. Preprocessing similarities

| Strategy                            | TabFM                                                     | TabICLv2/SDM                                                                         | Status and difference                                                                                     |
| ----------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Numeric mean imputation             | sklearn `SimpleImputer` inside `TransformToNumerical`     | [`MeanImpute`](sdm/processing/impute.py)                                             | Same core strategy; SDM is PyTorch-native                                                                 |
| Remove constant features            | `UniqueFeatureFilter`                                     | `ConstantFilter` is only a recipe placeholder                                        | TabFM-only today                                                                                          |
| Standard scaling                    | `CustomStandardScaler`                                    | [`StandardScale`](sdm/processing/standard_scale.py)                                  | Same population mean/std and epsilon concept; TabFM additionally hard-clips to `[-100, 100]`              |
| Two-stage 4-sigma soft clipping     | `OutlierRemover`                                          | [`SigmaClip`](sdm/processing/sigma_clip.py)                                          | Nearly line-for-line equivalent algorithm, including logarithmic soft bounds                              |
| Power normalization                 | sklearn Yeo–Johnson `PowerTransformer`                    | [`Power`](sdm/processing/power.py)                                                   | Same strategy; implementation and numerical edge behavior differ                                          |
| Quantile normalization              | sklearn `QuantileTransformer` / `RTDLQuantileTransformer` | [`Quantile`](sdm/processing/quantile.py)                                             | Same empirical-quantile strategy; TabFM's RTDL variant injects noise and follows it with standard scaling |
| Multiple normalization views        | `norm_methods` in `EnsembleGenerator`                     | `Choice` + `n_estimators` only described in the TabICLv2 recipe end-state            | Not implemented for TabICLv2 today                                                                        |
| Feature permutation                 | `FeatureShuffler(method="random")`                        | `FeaturePermute(method="latin")` placeholder                                         | Same ensemble-diversity purpose, different proposed schedule                                              |
| Class-label permutation             | cyclic `class_shift_offsets_`                             | `ClassShuffle(method="shift")` placeholder                                           | Same label-symmetry strategy; TabICLv2 side not implemented                                               |
| Classification temperature          | `softmax_temperature`, default 0.9                        | [`SoftmaxTemperature`](sdm/processing/postprocess.py), with 0.9 in the recipe sketch | Same `softmax(logits / temperature)` operation                                                            |
| Regression target scaling/inversion | sklearn `StandardScaler`                                  | `StandardScale` as target processor                                                  | Same strategy                                                                                             |

TabFM additionally handles categorical ordinal encoding, datetime expansion,
categorical-value permutation, feature and row subsampling, random feature
crosses, SVD features, OOF prediction, NNLS ensemble weighting, and probability
calibration. There is no implemented TabICLv2 counterpart for these features in
the current repository, so they should not be described as shared.

### 9. Public API, batching, checkpoint loading, and caching

The public wrappers share these behaviors:

- `fit` prepares context rather than optimizing model weights;
- `predict` combines stored context with unseen rows;
- classification and regression checkpoints are loaded separately from
  Hugging Face; and
- inference runs in eval/inference mode.

There are substantial API differences. TabFM exposes sklearn-compatible
estimators, NumPy/pandas preprocessing, ensemble members, per-member
`train_size`, mixed feature types, and padded heterogeneous batches.
`BaseModel` exposes a smaller PyTorch/Tensor API and uses the target length as a
single train/test boundary. TabICLv2's cache is integrated into the PyTorch
path; TabFM's full prefill/decode cache currently exists in the JAX path.

## TabFM sections that can be changed to TabICLv2-style code

This section uses three compatibility levels:

- **Safe**: can preserve the current model's intended output semantics and can
  preserve weights/outputs with a mechanical conversion and parity test.
- **Behavior-preserving architecture optimization**: should preserve the
  mathematical model, but requires implementation work and numerical tests.
- **Model variant**: still implements valid tabular ICL, but changes the learned
  function; current TabFM v1.0.0 checkpoints are not compatible and retraining
  is required.

### Safe or mechanically convertible substitutions

1. **Use an embedding for TabFM's ICL classification label encoder.**
   Replace `OneHotAndLinear` with `Embedding(max_classes, d_model)` and set
   `embedding.weight = projection.weight.T + projection.bias`. This is exact for
   valid class IDs. Preserve a separate unknown/sentinel embedding if invalid
   labels can reach this code.

2. **Use SDM's generic `Cache`/`KVCacheEntry` container around a TabFM PyTorch
   prefill/decode implementation.** The container is architecture-neutral.
   Cache both TabFM column stages and every ICL layer, retain per-example
   `train_size`, `d`, feature schema, and categorical masks, and do not replace
   TabFM's attention math. This changes cache plumbing, not the model.

3. **Use SDM's PyTorch-native preprocessing equivalents for the truly matching
   numeric steps.** `MeanImpute`, `StandardScale(epsilon=1e-6)`, and
   `SigmaClip(threshold=4)` can replace the corresponding numeric TabFM stages
   after parity tests. Add the explicit `[-100,100]` clip to match
   `CustomStandardScaler`; do not silently discard TabFM's categorical mask or
   dataframe encoders.

4. **Use TabICLv2's checkpoint-remapping pattern.** Fusing TabFM's separate
   q/k/v projections into one SDM projection is a mechanical concatenation when
   all head counts are unchanged. Parameter names and RoPE buffers can likewise
   be remapped. This is only safe if q/k RMSNorm, per-dimension scaling,
   RMSNorm/SwiGLU block order, biases, and output projections remain TabFM
   operations; replacing those equations is not a remap.

### Behavior-preserving architecture optimizations

1. **Query only CLS tokens in TabFM's final row block.** Mirror TabICLv2's
   `query=x[..., :K, :]` optimization in `row_interactor_2`'s final transformer
   block, while keeping the full sequence as keys/values. Since only CLS outputs
   are returned, the discarded final-layer feature-token outputs should be
   unobservable. Earlier blocks must still produce the full sequence.

2. **Query only test rows in TabFM's final ICL block.** TabICLv2 does this in
   `ICLBlock.forward`. It is valid when the caller only consumes test-row
   predictions and the last-layer context outputs are neither returned nor
   cached for another computation. TabFM's current full-sequence/OOF call paths
   must be audited before enabling it globally.

3. **Cache projected inducing K/V instead of raw inducing representations.**
   This mirrors `InducedTransformerBlock` and can avoid key/value projection in
   decode. It must be done per column stage and per set-transformer layer, with
   exact shape/padding parity against TabFM's JAX cache.

4. **Add a PyTorch prefill/decode API using TabICLv2's record/freeze/replay
   lifecycle.** Preserve TabFM's padded batch metadata and all three cache
   groups (`col1`, `col2`, and `icl`). The optimization is conceptually shared;
   copying only TabICLv2's one-column-stage cache would produce incorrect TabFM
   outputs.

### Model variants that require retraining

1. **Replace Fourier cell embedding with TabICLv2's grouped linear layer.**
   This removes separate categorical/numerical frequency banks and changes the
   input dimension. It is a valid simpler variant and is close to TabFM's JAX
   legacy mode, but current v1.0.0 weights cannot be reused.

2. **Adopt TabICLv2's `[1,2,4,…]` grouping offsets.** This changes which raw
   features every token sees and therefore requires retraining. Retain TabFM's
   `d`-aware wrapping and padded-column masking if heterogeneous ensemble
   batches remain supported.

3. **Collapse TabFM's two column/row cycles into one `RowEmbedding`.** This is
   the largest simplification: remove `col_embedder_2` and the full-output first
   row stage, then make the first/final row stage emit only readout tokens. It
   yields the TabICLv2 topology but invalidates the second-stage weights and
   changes representation depth.

4. **Replace PerDimScale/q-k RMSNorm with QASSMax.** QASSMax is particularly
   suitable for long contexts because it explicitly sees key count. Enable it
   in the inducing-to-context half of column attention and in dataset-wise ICL,
   matching TabICLv2; leave row attention on standard softmax. This changes
   attention logits and needs training or at least substantial fine-tuning.

5. **Replace RMSNorm/SwiGLU blocks with LayerNorm/GELU blocks.** This makes
   TabFM blocks TabICLv2-like, but changes parameters, residual dynamics, and
   checkpoint structure. It is not a source-only refactor.

6. **Change readout width from 8 × 256 to 4 × 128 (or any TabICLv2 default).**
   This changes the ICL model width and all downstream weights. The readout
   *strategy* is interchangeable; its dimensions are not.

7. **Replace scalar regression with a 999-quantile head.** Retrain with a
   quantile-appropriate objective and update every estimator path that assumes
   output dimension one. A compatibility wrapper can expose the 0.5 quantile as
   `predict()` while adding `predict_quantiles()` for the full output.

8. **Adopt TabICLv2's preprocessing/recipe abstraction wholesale.** This can
   keep the neural model valid, but it changes data distributions seen by the
   released checkpoint and drops TabFM features unless processors are first
   added for categorical data, datetime data, constant filtering, permutations,
   batching, augmentation, and ensemble combination. Treat this as a new
   evaluated model configuration, not a checkpoint-neutral cleanup.

## Changes that should not be made in isolation

- Do not remove `col_embedder_2` without also changing `row_interactor` output
  behavior and the ICL input width/path.
- Do not replace TabFM's Fourier embedder while loading its v1.0.0 checkpoint;
  the weight shapes and learned input basis differ.
- Do not substitute QASSMax for `PerDimScale` and claim checkpoint parity; the
  attention logits are different even if all projection weights are copied.
- Do not drop `d` and `cat_mask` merely because TabICLv2 does not expose them.
  TabFM's estimator batches ensemble members with padded feature counts and uses
  type-specific cell embeddings.
- Do not switch scalar regression to quantiles only in the decoder. The loss,
  output validation, inverse transform, ensembling, OOF predictions, NNLS
  fitting, and public API all depend on scalar output.
- Do not apply preprocessing learned on train + test rows. Both systems rely on
  context-only fitting to avoid leakage.
- Do not copy TabICLv2's cache keys literally into TabFM. TabFM needs two column
  caches plus ICL caches, and heterogeneous batches require associated lengths
  and masks.

## Recommended implementation order for a TabICLv2-style TabFM experiment

1. Build parity tests around the unmodified TabFM PyTorch/JAX outputs for
   classification and regression, including padded `d`, categorical masks,
   variable `train_size`, and prefill/decode.
2. Port only cache plumbing and final-query pruning first. These provide the
   clearest efficiency benefit without deliberately changing the learned model.
3. Add a configurable `cell_embedding={fourier,direct_grouped}` option and a
   configurable grouping offset schedule. Train the direct-grouped variant from
   scratch.
4. Add `attention_scaling={per_dim,qassmax}` while preserving the remainder of
   the TabFM block, so QASSMax can be ablated independently.
5. Add `num_column_row_cycles={1,2}` and train both settings with otherwise
   matched dimensions.
6. Add a quantile regression decoder only after the estimator supports an
   explicit distributional output contract.

Minimum validation should cover output shape, no test-label leakage,
permutation behavior, padding invariance, cache vs non-cache parity, JAX vs
PyTorch parity where applicable, mixed categorical/numeric inputs, and
classification/regression estimator integration.

## Bottom line

TabICLv2 and TabFM are close relatives at the level of processing strategy:
shifted feature groups, early target conditioning, induced column attention,
readout-token row aggregation, label-conditioned dataset attention, and
context caching all correspond directly. The most reusable TabICLv2 ideas for
current TabFM without retraining are cache organization, projected-K/V reuse,
and final-layer query pruning. Direct grouped embedding, QASSMax, a single
column/row cycle, LayerNorm/GELU blocks, smaller readout dimensions, and
quantile regression are coherent TabICLv2-style TabFM variants, but they are
new models and must not be presented as compatible with released TabFM v1.0.0
weights.
