# NanoTabICL

The [`sdm.nn`](api/nn) package provides reusable, GPU-accelerated building blocks for developing foundation models for structured data from scratch.
In this guide, we show how to build a minimal [TabICLv2](https://arxiv.org/abs/2602.11139)-inspired tabular foundation model (TFM) with it.
Like other [in-context learning (ICL)](icl) models, it receives labeled context rows together with unlabeled query rows and predicts the query targets in a single forward pass.

[TabICL](https://arxiv.org/abs/2502.05564) pioneered the **"Compress-then-ICL"** approach, which has since been adopted by other TFMs such as [TabPFN-3](https://arxiv.org/abs/2605.13986), [TabFM](https://arxiv.org/abs/2609.37959), and [KumoTabular](https://huggingface.co/blog/nvidia/kumo-tabular).
Its main advantage is its efficient ICL stage that operates on compressed row representations rather than individual cells.
It consists of three stages:

```{figure} images/nanotabicl_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/nanotabicl_dark.svg
:figclass: dark-only
:width: 100%
```

1. A **row-wise compression block** that maps each table row to a fixed-size embedding.
2. An **in-context learning block** that models interactions among labeled context rows and uses that context to update query representations.
3. A final **prediction head**.

## Row-wise Compression

The compression block embeds each table row using column-wise attention followed by row-wise attention.
The column-wise attention captures patterns within each column.
Afterwards, row-wise attention combines cell information within each row into a single fixed-size embedding.

### Column-wise Attention

Since self-attention grows quadratically with the number of cells, full self-attention across an entire table is not scalable.
Thus, the column-wise Transformer uses induced self-attention from the [Set Transformer](https://arxiv.org/abs/1810.00825).
The key innovation of induced self-attention is to learn a small, fixed number of inducing points.
These points first attend to the column entries.
Then, column entries attend to the induced representation.
This two-stage mechanism thus replaces direct attention between all pairs of elements.
The inducing points are learnable parameters.
During each forward pass, they attend only to context entries; both context and query entries then attend to the resulting representation.
This prevents queries within a batch from influencing each other and allows caching across repeated predictions.

The {py:class}`sdm.nn.InducedTransformerBlock` provides a highly reusable and configurable block for induced self-attention, built from two {py:class}`sdm.nn.TransformerBlock` modules: **(1)** the `inducing_block` that updates the learned inducing points from context entries, and **(2)** the `output_block` that updates all column entries from the induced representation.

We start by defining a simple feed-forward multilayer perceptron (MLP):

```python
import torch
from torch import Tensor

def MLP(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.LayerNorm(in_channels),
        torch.nn.Linear(in_channels, 2 * in_channels),
        torch.nn.GELU(),
        torch.nn.Linear(2 * in_channels, out_channels),
    )
```

Next, our `ColumnTransformer` stacks three layers of column-wise induced attention over 128 inducing points.
Its input is a `D`-dimensional cell embedding `x` for each row `R` and each column `C`.
The first `y.size(0)` rows of `x` are context rows with corresponding labels in `y`; the remaining rows are query rows.

TabICLv2 uses a pre-norm {py:class}`~sdm.nn.TransformerBlock`, applying {py:class}`~torch.nn.LayerNorm` before each attention and feed-forward sublayer.
When attending over context rows, TabICLv2 uses {py:class}`sdm.nn.QASSMax` to scale queries based on the number of context rows, adapting attention to varying context sizes.
The {py:class}`~sdm.nn.TransformerBlock` supports these attention patterns through configurable normalization, query scaling, and separate query and key/value inputs.

Lastly, to produce task-conditioned compressed representations (*e.g.*, to filter out task-irrelevant columns), TabICLv2 injects target information from context examples before entering the Transformer.
This is done by adding a learnable class embedding to each cell in the context rows, supporting up to 10 classes.

```{note}
We drop the leading batch dimension in all tensor shapes for simplicity.
In practice, TFMs are trained in batches.
```

```python
import sdm

class ColumnTransformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.class_emb = torch.nn.Embedding(10, channels)

        self.blocks = torch.nn.ModuleList([
            sdm.nn.InducedTransformerBlock(
                channels=channels,
                num_inducing_points=128,
                inducing_block=sdm.nn.TransformerBlock(
                    channels=channels,
                    num_query_heads=num_heads,
                    query_norm=torch.nn.LayerNorm(channels),
                    key_value_norm=torch.nn.LayerNorm(channels),
                    mlp=MLP(channels, channels),
                    query_scaling=sdm.nn.QASSMax(
                        channels=channels // num_heads,
                        num_heads=num_heads,
                    ),
                ),
                output_block=sdm.nn.TransformerBlock(
                    channels=channels,
                    num_query_heads=num_heads,
                    query_norm=torch.nn.LayerNorm(channels),
                    key_value_norm=torch.nn.LayerNorm(channels),
                    mlp=MLP(channels, channels),
                ),
            ) for _ in range(3)
        ])

    def forward(
        self,
        x: Tensor,  # [R, C, D]
        y: Tensor,  # [R_context]
    ) -> Tensor:  # [R, C, D]

        # Inject task labels into context cell representations:
        x = torch.cat([
            x[:y.size(0)] + self.class_emb(y)[:, None, :],
            x[y.size(0):],
        ], dim=0)

        # Use the row dimension as the sequence axis:
        x = x.transpose(0, 1)  # [C, R, D]
        for block in self.blocks:
            # Inducing points should only attend to context rows:
            x = block(query=x, key_value=x[:, :y.size(0)])
        return x.transpose(0, 1)  # [R, C, D]
```

### Row-wise Attention

We next perform self-attention across the column vectors of each row.
As the number of columns in a dataset is usually much smaller than the number of rows, we use a full self-attention {py:class}`~sdm.nn.TransformerBlock`.

TabICLv2 prepends four learnable `[CLS]` tokens to each row and uses their output representations as the row's final embedding.
This maps rows with varying numbers of columns to fixed-size embeddings.
The `[CLS]` tokens are initialized from a truncated normal distribution and refined via repeated row-wise attention.
Using multiple `[CLS]` tokens allows more information to pass through this bottleneck.
Their four output vectors are flattened into a `4*D`-dimensional row embedding, which becomes a single token in the ICL stage.

Furthermore, TabICLv2 applies {py:class}`sdm.nn.RotaryEmbedding` (RoPE) in row-wise attention to distinguish features by their column positions.
This helps resolve ambiguity between features with similar representations.

```python
class RowTransformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.cls_tokens = torch.nn.Parameter(torch.empty(4, channels))
        torch.nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        rope = sdm.nn.RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
        )
        self.blocks = torch.nn.ModuleList([
            sdm.nn.TransformerBlock(
                channels=channels,
                num_query_heads=num_heads,
                query_norm=torch.nn.LayerNorm(channels),
                key_value_norm=torch.nn.LayerNorm(channels),
                query_transform=rope,
                key_transform=rope,
                mlp=MLP(channels, channels),
            ) for _ in range(3)
        ])

    def forward(
        self,
        x: Tensor,  # [R, C, D]
    ) -> Tensor:  # [R, 4*D]

        # Prepend [CLS] tokens:
        cls = self.cls_tokens.expand(x.size(0), -1, -1)  # [R, 4, D]
        x = torch.cat([cls, x], dim=1)  # [R, 4 + C, D]

        for block in self.blocks:
            x = block(x)
        return x[:, :4, :].flatten(1)  # [R, 4*D]
```

Together, the column-then-row attention blocks form the row-wise compression block of our tabular foundation model.

## In-Context Learning

The ICL stage operates on compressed row representations, modeling interactions among rows through a sequence of {py:class}`~sdm.nn.TransformerBlock` layers:

1. Query rows attend only to context rows, so each prediction is independent of other queries.
2. Context rows attend only to context rows, keeping their representations independent of queries.

Unlike `ColumnTransformer`, TabICLv2's ICL stage uses full self-attention over context rows.
Before attention, it again adds target embeddings to context row representations.

```python
class ICLTransformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.class_emb = torch.nn.Embedding(10, channels)

        self.blocks = torch.nn.ModuleList([
            sdm.nn.TransformerBlock(
                channels=channels,
                num_query_heads=num_heads,
                query_norm=torch.nn.LayerNorm(channels),
                key_value_norm=torch.nn.LayerNorm(channels),
                query_scaling=sdm.nn.QASSMax(
                    channels=channels // num_heads,
                    num_heads=num_heads,
                ),
                mlp=MLP(channels, channels))
            for _ in range(12)
        ])

    def forward(
        self,
        x: Tensor,  # [R, 4*D]
        y: Tensor,  # [R_context]
    ) -> Tensor:  # [R_query, 4*D]

        # Reinject task labels into context row representations:
        x = torch.cat(
              [x[:y.size(0)] + self.class_emb(y),
              x[y.size(0):]],
              dim=0,
          )

        for block in self.blocks:
            x = block(query=x, key_value=x[:y.size(0)])
        return x[y.size(0):]  # [R_query, 4*D]
```

After the ICL module, an MLP prediction head maps each query row representation to a fixed number of class logits.

## Putting It All Together

We can now compose the blocks into a tabular foundation model for classification, made up of a `ColumnTransformer`, `RowTransformer`, and `ICLTransformer` in sequence:

```python
class NanoTabICL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cell_emb = torch.nn.Linear(1, 128)
        self.column_transformer = ColumnTransformer(128, num_heads=8)
        self.row_transformer = RowTransformer(128, num_heads=8)
        self.icl_transformer = ICLTransformer(4*128, num_heads=8)
        self.head = MLP(4*128, 10)

    def forward(self,
        x: Tensor,  # [R, C]
        y: Tensor,  # [R_context]
    ) -> Tensor:  # [R_query, 10]

        x = self.cell_emb(x.unsqueeze(-1))  # [R, C, D]
        x = self.column_transformer(x, y)  # [R, C, D]
        x = self.row_transformer(x)  # [R, 4*D]
        x = self.icl_transformer(x, y)  # [R_query, 4*D]
        x = self.head(x)

        return x
```

The model takes numerical floating-point features `x` of a table with shape `[R, C]` and integer labels `y` for the first `y.size(0)` context rows.
The remaining rows are treated as query rows whose labels the model predicts.
Categorical encoding, missing-value handling, and feature scaling are typically performed in a preprocessing stage before the model's {py:meth}`~torch.nn.Module.forward` call.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(128, 32, device=device)
y = torch.randint(0, 10, (64, ), device=device)

model = NanoTabICL().to(device)
out = model(x, y)
```
