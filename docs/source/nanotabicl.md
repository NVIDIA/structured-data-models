# Building a Tabular Foundation Model

This guide builds a minimal TabICL-inspired tabular foundation model, `NanoTabICL`, by composing building blocks from the `sdm` library.
Like other in-context learning models, `NanoTabICL` receives labeled *context* rows together with unlabeled *query* rows, and predicts the query targets in a single forward pass without updating its weights (see the [In-Context Learning](icl) guide).
The architecture presented here should not be used for training, since several simplifications have been made to present the core ideas - some of them are discussed in the [Extensions](extensions) section.
We drop the leading batch dimension in all tensor shapes for simplicity.
In practice, tabular foundation models are trained in batches.

The model is a stack of three stages:

1. An encoder that turns every table row into a fixed-size embedding (a column-wise, then a row-wise transformer).
2. An in-context learning (ICL) transformer that lets query rows learn from labeled context rows.
3. A prediction head.

`NanoTabICL` is modeled after {py:class}`~sdm.models.TabICLv2` ([Qu et al., 2026](https://arxiv.org/abs/2602.11139)).

## The MLP Helper

The following simple feed-forward multilayer perceptron (MLP) helper function keeps the layer definitions brief in the sections below:

```python
from typing import Any

import sdm
import sdm.nn

import torch
from torch import nn
from torch import Tensor


def mlp(channels: int, out_channels: int | None = None) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(channels),
        nn.Linear(channels, 2 * channels),
        nn.GELU(),
        nn.Linear(2 * channels, out_channels if out_channels is not None else channels)
    )
```

## Encoder

The encoder embeds each table row, and consists of a column-wise transformer followed by a row-wise transformer.
Its output can be interpreted as one embedding per row.

### Column Transformer

The first encoder stage is a column-wise set transformer, which operates on one embedding per table cell.

Set transformers ([Lee et al., 2018](https://arxiv.org/abs/1810.00825)) are order-invariant transformers, which is desirable because predictions should not depend on row permutations.
A naive set transformer would remove any sequential masking and positional encoding from the classic transformer architecture.
Since self-attention grows quadratically with set size, however, full self-attention across all cells of a column would not be scalable.

The key innovation of set transformers is to learn a low, fixed number of inducing points.
These points first attend to the set elements, before the set elements attend to the induced output.
The inducing points themselves are learnable parameters.
A set transformer can be seen as a compressed version of full self-attention.

Induced self-attention blocks are made up of two sequential transformer blocks:

1. The inducing block: the inducing points are learned by attending to the input set, in this case the column values.
2. The output block: the input set (the column values) acts as the query and attends to the inducing points' output from step 1, so every cell reads a compressed summary of its whole column instead of attending to every other cell.

The column stage is the first place where information flows between rows.
We therefore restrict the keys and values of the inducing block to the context rows, while the output block still runs for every row, context or query.
This prevents queries within a batch from influencing each other, and allows caching across query rows (see [KV Caching](kv-caching)).

{py:class}`~sdm.nn.InducedTransformerBlock` provides a highly reusable and configurable block for induced self-attention, built from two {py:class}`~sdm.nn.TransformerBlock` modules.

Tensor dimension key:

- `R`: the number of rows.
- `C`: the number of columns.
- `d`: the embedding dimension.

```python
class ColumnTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_inducing_points: int,
        num_heads: int,
        num_layers: int,
        inducing_block_kw: dict[str, Any] | None = None,
        output_block_kw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            sdm.nn.InducedTransformerBlock(
                channels=channels,
                num_inducing_points=num_inducing_points,
                inducing_block=sdm.nn.TransformerBlock(
                    channels, num_heads, mlp(channels), **(inducing_block_kw or {})
                ),
                output_block=sdm.nn.TransformerBlock(
                    channels, num_heads, mlp(channels), **(output_block_kw or {})
                ),
            ) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, n_ctx: int) -> Tensor:
        # x: (R, C, d), where the first n_ctx rows are context rows and the rest are query rows
        x = x.transpose(0, 1)  # (C, R, d): since transformer block attends over second last dimension
        for block in self.blocks:
            x = block(x, key_value=x[:, :n_ctx])  # inducing points only attend to context rows
        return x.transpose(0, 1)  # (R, C, d)
```

The learned inducing vectors of each block are available through its `inducing_points` property.

### Row Transformer

Since TabICL uses column-wise, then row-wise attention, we next perform self-attention across the column vectors of each row.
As the number of columns in a dataset is usually much smaller than the number of rows, we can use a full self-attention block ({py:class}`~sdm.nn.TransformerBlock`) here.

The number of column and row transformer blocks can be repeated.
In TabICLv2, 3 column attention layers are stacked, followed by 3 row attention layers.
In {py:class}`~sdm.models.KumoTabular`, the column and row attention blocks are interleaved.

The final row transformer block is the embedding layer before the ICL module.
Hence, it must output a fixed-size tensor that is independent of the number of table columns, in order to decouple the ICL stage from the table size.
To achieve this, we aggregate each row into a fixed number of learned "readout" vectors (also called CLS tokens, borrowing from the NLP literature).
This is similar to the first half of a set transformer, in that we compress to a fixed set of vectors.
However, here we have full self-attention at each layer, and only read out the CLS tokens of the last layer.
We default to 4 readout tokens, as TabICL does.

Together, the column-then-row attention blocks form the encoder of our tabular foundation model.

The readout tokens are initialized from a truncated normal distribution, to break the symmetry between readout tokens.
Ultimately, they are concatenated into a single, large vector for the ICL stage.
Using multiple readout tokens allows more information to pass through this bottleneck.

Tensor dimension key:

- `R`: the number of rows.
- `C`: the number of columns.
- `M`: the number of readout tokens.
- `d`: the embedding dimension.

```python
class RowTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        num_readout_tokens: int = 4,
        **transformer_block_kw,
    ) -> None:
        super().__init__()
        self.readout_tokens = nn.Parameter(torch.empty(num_readout_tokens, channels))
        nn.init.trunc_normal_(self.readout_tokens, std=0.02)

        self.blocks = nn.ModuleList([
            sdm.nn.TransformerBlock(channels, num_heads, mlp(channels), **transformer_block_kw)
            for _ in range(num_layers)
        ])

    @property
    def num_readout_tokens(self) -> int:
        return self.readout_tokens.size(0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (R, C, d)
        readout = self.readout_tokens.expand(x.size(0), -1, -1)  # (R, M, d)
        x = torch.cat([readout, x], dim=1)  # (R, M + C, d)
        for block in self.blocks:
            x = block(x)
        return x[:, :self.num_readout_tokens, :]  # (R, M, d)
```

## In-Context Learning Block

Now that we have embeddings for each context and query row from the encoder, we add an in-context learning stage.
This is again a sequence of {py:class}`~sdm.nn.TransformerBlock` modules, with one key mechanism: the attention mask, which follows two rules:

1. Query rows may only attend to context rows.
2. Context rows may attend to each other, but not to query rows.

Rule 1 ensures that predictions are stable across batches: they do not depend on which other queries were batched together, and information cannot leak between queries.
Rule 2 ensures that the context computation is independent of the queries, and so can be cached across a large number of query rows (see [KV Caching](kv-caching)).

Note that, unlike in the encoder stages, we pay the computational price of full self-attention over the context rows here.
This scales quadratically in the number of rows, but in practice it proves its worth empirically.
Each row is represented by a single token, so there is no additional complexity factor from the number of columns, unlike in the encoder.

Tensor dimension key:

- `R_ctx`: the number of context rows.
- `R_query`: the number of query rows.
- `M`: the number of readout tokens.
- `d`: the embedding dimension.

```python
class ICLTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            sdm.nn.TransformerBlock(channels, num_heads, mlp(channels))
            for _ in range(num_layers)
        ])

    def forward(self, x_context: Tensor, x_query: Tensor) -> Tensor:
        # x_context: (R_ctx, M * d) label-aware context row embeddings
        # x_query: (R_query, M * d)
        n_ctx = x_context.size(0)

        x = torch.cat([x_context, x_query], dim=0)  # (R_ctx + R_query, M * d)

        # attn_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        # attn_mask[:n_ctx] = True

        for block in self.blocks:
            x = block(query=x, key_value=x[:n_ctx])

        return x[n_ctx:]  # (R_query, M * d)
```

## Prediction Head

After the ICL module, the model needs a way to make predictions.
The head is just an MLP from the ICL output to `out_channels`, so it is not specific to classification: setting `out_channels` to the number of classes gives class logits.
This is the `mlp` helper from above, whose leading layer norm also serves as the final normalization before the head.
For classification, a softmax over the logits gives the class probabilities.

## Putting It Together

We can now compose the blocks into a tabular foundation model for classification, made up of three stages in sequence:

1. The encoder: each cell is first embedded with a shared linear layer, which assumes standardized numerical features, followed by a column-wise, then a row-wise transformer. The output is flattened so that the readout tokens form one larger vector representation of each row, and the embedding scale is normalized to stabilize training in the ICL layers. In TabICL and other architectures, each transformer block also normalizes its attention inputs, but we omit this for simplicity.
2. The ICL module: a transformer stack in which query rows learn from context rows.
3. The prediction head: a classification head that maps the ICL output to class logits.

Labels enter the model in two places, both added to the context rows only: once to the cell embeddings before the column stage, and once to the flattened row embeddings before the ICL stage.
This is how the model learns from the context: context rows carry their targets as label embeddings, while query rows do not, so the model has to infer the query targets from the context.

Tensor dimension key:

- `R`: the number of rows.
- `C`: the number of columns.
- `R_ctx`: the number of context rows.
- `R_query`: the number of query rows.
- `M`: the number of readout tokens.
- `d`: the embedding dimension.

```python
class NanoTabICL(nn.Module):
    """A small, TabICL inspired tabular foundation model."""

    def __init__(
        self,
        channels: int,
        num_readout_tokens: int = 4,
        num_classes: int = 2,
        num_heads: int = 4,
        num_layers: int = 1,
        num_inducing_points: int = 16,
    ) -> None:
        super().__init__()
        icl_channels = num_readout_tokens * channels

        self.cell_embedding = nn.Linear(1, channels)  # shared across all cells
        self.column_label_embedding = nn.Embedding(num_classes, channels)
        self.column_transformer = ColumnTransformer(
            channels=channels,
            num_inducing_points=num_inducing_points,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.row_transformer = RowTransformer(
            channels=channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_readout_tokens=num_readout_tokens,
        )
        self.encoder_norm = nn.LayerNorm(channels)
        self.icl_label_embedding = nn.Embedding(num_classes, icl_channels)
        self.icl_transformer = ICLTransformer(
            channels=icl_channels,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.prediction_head = mlp(icl_channels, out_channels=num_classes)

    def forward(self, x: Tensor, y_context: Tensor) -> Tensor:
        # x: (R, C) numerical features, y_context: (R_ctx,) integer class labels of the context rows
        n_ctx = y_context.size(0)
        x = self.cell_embedding(x.unsqueeze(-1))  # (R, C, d)

        # Label-aware cell embeddings: context rows only
        x_context = x[:n_ctx] + self.column_label_embedding(y_context).unsqueeze(1)  # (R_ctx, C, d)
        x = torch.cat([x_context, x[n_ctx:]], dim=0)  # (R, C, d)

        x = self.column_transformer(x, n_ctx)
        x = self.encoder_norm(self.row_transformer(x)).flatten(start_dim=1)  # (R, M * d)

        # Label-aware row embeddings: context rows only
        x_context = x[:n_ctx] + self.icl_label_embedding(y_context)  # (R_ctx, M * d)
        x_query = self.icl_transformer(x_context, x[n_ctx:])  # (R_query, M * d)
        return self.prediction_head(x_query)  # (R_query, num_classes)
```

The defaults use a single layer of each kind, whereas TabICLv2 stacks 3 column layers, 3 row layers and 12 ICL layers.
The number of heads must divide both `channels` and `num_readout_tokens * channels`.

(extensions)=

## Extensions

The following are some extra mechanisms included in TabICLv2 which were not covered above.

### Feature Grouping

An improvement that TabICLv2 introduces is feature grouping as the first step of the encoder.

Since columns are embedded first in TabICL, columns with similar distributions can collapse into the same embedding, which makes it hard for the model to distinguish them.
TabICLv2 therefore linearly combines groups of columns, so that pairs of columns are never in the same group.

A simple grouping that guarantees no overlapping pairs between groups is the following:

- For each column _i_, group it with columns _i+1_ and _i+3_, wrapping around indices if needed.
- No two groups share 2 columns, due to the unique differences between offsets.

In general, using shift offsets of `(0, 1, 3, ..., 2**n - 1)`, which form groups of `n + 1` columns, guarantees no overlap when there are at least `2**(n+1) - 1` columns.
In practice, the TabICL authors found no major performance benefit from groups larger than 3, so we'll default to `(0, 1, 3)`.

After grouping, we apply a shared linear transformation that maps all groups to a fixed embedding dimension, ready for the next layer.
This generalizes the cell embedding of `NanoTabICL` from single columns to groups of columns, so using it amounts to replacing `self.cell_embedding(x.unsqueeze(-1))` in `NanoTabICL` with a `FeatureGrouping(embedding_dim=channels)` applied to `x`.

Tensor dimension key:

- `R`: the number of rows.
- `C`: the number of columns.
- `d`: the embedding dimension.

```python
class FeatureGrouping(nn.Module):
    """A simplified feature grouping mechanism for single tables, inspired by TabICLv2."""

    def __init__(
        self,
        shift_pattern: tuple[int, ...] = (0, 1, 3),
        embedding_dim: int = 64
    ) -> None:
        super().__init__()
        self.shift_pattern = shift_pattern
        self.lin = nn.Linear(len(shift_pattern), embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2  # (R, C)

        # accumulate into (R, C, #offsets)
        groups = torch.stack(
            [x.roll(offset, dims=1) for offset in self.shift_pattern],
            dim=-1,
        )

        return self.lin(groups)  # (R, C, d)
```

### QASSMax

Softmax attention spreads out as the number of keys grows: each key gets a smaller weight, and attention can no longer focus on a few relevant rows.
This matters for tabular in-context learning, where the context can range from a handful to many thousands of rows.

Query-Aware Scalable SoftMax ({py:class}`~sdm.nn.QASSMax`), introduced in TabICLv2 ([Qu et al., 2026](https://arxiv.org/abs/2602.11139)), overcomes this by rescaling the query depending on the number of context rows and on the query itself.

(kv-caching)=

### KV Caching

The same context is often used to predict many batches of query rows, and since the context is usually much larger than a query batch, recomputing it for every batch dominates the cost of prediction.

The attention masks in our model ensure that context rows never attend to query rows.
The keys and values that the context produces in each layer can therefore be computed once and reused for every batch of query rows - this is what {py:class}`~sdm.cache.Cache` does.
ICL models use it under the hood: {py:meth}`~sdm.models.ICLModel.fit` caches the context once, and each {py:meth}`~sdm.models.ICLModel.predict` call reuses it.
