# TabFM integration in Structured Data Models

## Summary

This change adds a native PyTorch implementation of TabFM to
`structured-data-models` (SDM). The implementation is designed to load the
released TabFM v1.0.0 PyTorch checkpoints without remapping model parameters,
while presenting the same tensor-oriented `forward`, `fit`, and `predict`
workflow used by other SDM model families.

The integration is intentionally limited to reusable model, preprocessing,
checkpoint, and caching foundations. It does not add the upstream sklearn
estimator, ensemble orchestration, platform APIs, or serving abstractions.

Detailed implementation history is available in
[`TABFM_SDM_INTEGRATION_LOG.md`](TABFM_SDM_INTEGRATION_LOG.md). The original
implementation procedure and architecture analysis are in
[`TABFM_SDM_INTEGRATION_PLAN.md`](TABFM_SDM_INTEGRATION_PLAN.md) and
[`TABICLV2_TABFM_COMPARISON.md`](TABICLV2_TABFM_COMPARISON.md).

## Why TabFM has model-specific neural blocks

TabFM and TabICLv2 share a high-level tabular in-context learning strategy:

1. group and embed feature values;
2. inject targets into context rows;
3. summarize columns with induced attention;
4. combine feature tokens within each row;
5. attend from query rows to labeled context rows; and
6. reuse context state across repeated predictions.

Their checkpoint-level equations are not interchangeable. Released TabFM
weights depend on:

- learned Fourier cell embeddings and `[0, 1, 3]` group offsets;
- separate query, key, value, and output projections;
- per-head query/key RMS normalization;
- learned positive per-dimension query scaling;
- RMSNorm and SwiGLU transformer blocks;
- interleaved RoPE with checkpoint-loaded frequency buffers;
- two column/row interaction cycles; and
- scalar regression output.

Replacing these with TabICLv2's grouped linear embedding, fused projections,
QASSMax, LayerNorm/GELU blocks, split-half RoPE, single column/row cycle, or
quantile head would change the learned function. Those alternatives therefore
remain separately trained model variants rather than integration refactors.

## Shared SDM components

The integration reuses shared modules where their contracts are genuinely
compatible:

- `BaseModel` supplies the common SDM model boundary.
- `TableTensor` preserves numerical and categorical columns at the input
  boundary.
- `Cache` and `KVCacheEntry` provide the record/freeze/replay lifecycle.
- `Recipe`, `MeanImpute`, `StandardScale`, `Clamp`, `SigmaClip`, and `Identity`
  provide tensor-native preprocessing.
- `SDPA` performs the final attention kernel after TabFM applies its own
  checkpoint-compatible projection, normalization, and scaling operations.
- `apply_rotary_embedding` supplies shared split-half and interleaved rotary
  tensor operations. TabFM retains its own wrapper so the official `freqs`
  buffer name and layout remain unchanged.

Shared `SDPA` gained an optional explicit scale so TabFM can use `scale=1.0`.
Its default scaling remains unchanged for existing SDM models. It also accepts
boolean and additive masks.

The generic stateless `Clamp` processor was added because both TabFM and
TabICLv2 require fixed `[-100, 100]` clipping after standardization. This
avoids duplicating model-specific clipping code.

## Package structure

```text
sdm/models/tabfm/
├── __init__.py          public exports
├── attention.py         checkpoint-compatible attention and encoders
├── checkpoint.py        strict local checkpoint loading
├── embedding.py         cell and induced column embeddings
├── icl.py               dataset-wise in-context learning
├── mlp.py               checkpoint-compatible target/decoder MLP
├── model.py             TabFMCore assembly and public TabFM wrapper
├── recipe.py            classification and regression recipes
└── row_interaction.py   feature-wise row interaction and CLS readout
```

`TabFMCore` mirrors the released neural model and returns every row for parity
testing. The public `TabFM` wrapper returns query rows through the SDM API and
dispatches to classification or regression based on target dtype.

## Public usage

Randomly initialized models require no external assets:

```python
import torch

from sdm.models import TabFM

model = TabFM(pretrained=False)
features = torch.randn(1, 12, 5)
context_target = torch.randint(0, 3, (1, 8))
query_logits = model(features, context_target)
```

Context state can be reused:

```python
model.fit(features[:, :8], context_target)
first_logits = model.predict(features[:, 8:10])
second_logits = model.predict(features[:, 10:])
```

Integer targets select classification. Floating-point targets select scalar
regression.

## Checkpoints and licensing

TabFM source code is Apache 2.0, and adapted source files retain the applicable
copyright, license, and modification notices.

Released TabFM weights use the TabFM Non-Commercial License v1.0. The package
does not download or redistribute them. Pretrained loading requires an
explicitly obtained local directory:

```text
checkpoint/
├── classification/
│   ├── config.json
│   └── model.safetensors
└── regression/
    ├── config.json
    └── model.safetensors
```

Loading is strict, supports an explicit dtype/device conversion, and has no
network fallback. `pretrained=False` performs no checkpoint access.

## Preprocessing

The default numerical feature pipeline is:

```text
MeanImpute
  -> StandardScale(epsilon=1e-6)
  -> Clamp(-100, 100)
  -> SigmaClip(threshold=4)
```

Regression targets are standardized and inverted after prediction.
Classification targets remain integer class IDs. Learned preprocessing state
must be fitted only on context rows. Categorical indices and their mask bypass
the numerical pipeline.

Advanced upstream estimator behavior—multiple normalization views, feature or
class permutations, SVD/cross augmentation, OOF prediction, NNLS weighting,
and calibration—is intentionally outside this lightweight core integration.

## Projected context cache

`TabFM.fit()` records:

- projected inducing keys and values for both column stages;
- projected context keys and values for every ICL block; and
- schema, batch, categorical-routing, active-width, device, and dtype metadata.

It does not retain raw context features or targets. `TabFM.predict()` validates
the metadata and evaluates only query rows against the frozen projected state.
Column cache entries remain compatible with `col_chunk_size`, preserving the
model's bounded-memory execution strategy.

## Validation

The integration includes component, upstream-parity, wrapper, recipe,
checkpoint, cache, compilation, and hardware-gated tests. The current local
baseline is:

- TabFM suite: `154 passed, 16 skipped`;
- full repository: `367 passed, 90 skipped`;
- warnings-as-errors Sphinx build: passed; and
- Ruff, Ruff format, `ty`, dependency, and whitespace checks: passed.

The TabFM skips are explicit: 14 require CUDA and two require an appropriately
licensed local official-checkpoint directory. Actual official-weight parity
and CUDA peak-memory measurements remain environment-dependent release checks,
as tracked in [`TABFM_SDM_NEXT_STEPS.md`](TABFM_SDM_NEXT_STEPS.md).

## Suggested review areas

Reviewers should focus on:

1. preservation of upstream parameter names and operation order;
2. context/query masking and absence of query-target leakage;
3. categorical routing and padded active feature widths;
4. strict, local-only checkpoint handling and license messaging;
5. cache equivalence with uncached inference;
6. context-only preprocessing state; and
7. whether shared SDM utilities remain generic and backward compatible.
