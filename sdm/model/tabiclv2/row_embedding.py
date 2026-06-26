"""Row embedding module for structured tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    Embedding,
    LayerNorm,
    Linear,
    ModuleList,
    Parameter,
)

from sdm.nn import (
    InducedTransformerBlock,
    RotaryEmbedding,
    TransformerBlock,
)
from sdm.task import TaskType


class RowEmbedding(torch.nn.Module):
    r"""Encode a table into per-row embeddings via induced set attention.

    Each table is a tensor of shape ``[B, R, C]`` holding ``R`` rows of ``C``
    feature columns for ``B`` tables. The first ``R_train`` rows additionally
    carry a target ``y`` that is embedded and added to the row features.

    The module applies two attention stages in sequence:

    * **Column-wise (:class:`~sdm.nn.InducedTransformerBlock`).** Treating each
      column as an independent set of rows, a stack of induced transformer
      blocks lets a small set of learnable inducing points attend to the
      context (training) rows, after which the full set of rows attends back to
      those inducing points. This reduces the per-column cost from
      :math:`O(R^2)` to :math:`O(R \cdot m)` for ``m`` inducing points.
    * **Row-wise.** Each row attends across its columns (and the prepended
      readout tokens) with rotary positional embeddings. The final layer
      reads out only the readout tokens.

    Restricting the inducing points to attend over the first ``R_train`` rows
    keeps target information from leaking into the row representations.

    Args:
        task_type: A :class:`~sdm.TaskType` selecting how the target ``y`` is
            embedded. ``classification`` looks up a learned embedding per
            class index (via :class:`torch.nn.Embedding`), while
            ``regression`` linearly projects the scalar target into
            ``channels`` dimensions (via a bias-free :class:`torch.nn.Linear`).
        channels: The per-token channel count ``D``.
        num_layers: The number of column/row attention layers.
        num_heads: The number of attention heads.
        group_size: The number of columns ``G`` grouped into each token.
        num_inducing_points: The number of learnable inducing points ``m``.
        num_readout_tokens: The number of readout tokens ``K`` prepended before
            row-wise attention.
        num_classes: The number of target classes for classification, i.e. the
            number of embeddings in the target lookup table. Unused for
            regression.
        feedforward_channels: The hidden width of the MLP in each attention
            block. Defaults to ``2 * channels`` when ``None``.
        qassmax: Whether the column-wise
            :class:`~sdm.nn.InducedTransformerBlock` layers use
            :class:`~sdm.nn.QASSMax`.
        norm_bias: Whether LayerNorm layers use a learnable bias.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.
    """

    def __init__(
        self,
        task_type: TaskType,
        channels: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        group_size: int = 3,
        num_inducing_points: int = 128,
        num_readout_tokens: int = 4,
        num_classes: int = 10,
        feedforward_channels: int | None = None,
        qassmax: bool = True,
        norm_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        if feedforward_channels is None:
            feedforward_channels = 2 * channels

        self.task_type = task_type

        self.lin = Linear(group_size, channels, **factory_kwargs)

        if self.task_type == TaskType.classification:
            self.y_emb: torch.nn.Module = Embedding(
                num_embeddings=num_classes,
                embedding_dim=channels,
                **factory_kwargs,
            )
        else:
            self.y_lin = Linear(1, channels, bias=False, **factory_kwargs)

        self.col_layers = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                num_inducing_points=num_inducing_points,
                qassmax=qassmax,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.readout_token = Parameter(
            torch.empty((1, 1, num_readout_tokens, channels), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        self.row_layers = ModuleList(
            TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                qassmax=False,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.rope = RotaryEmbedding(
            channels=channels // num_heads,
            theta=100_000,
            **factory_kwargs,
        )

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [B, R, C]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R, K * D]
        r"""Encode a batch of tables into per-row embeddings.

        Args:
            x: Feature tensor with shape ``[B, R, C]`` for ``B`` tables, ``R``
                rows, and ``C`` columns.
            y: Target values for the first ``R_train`` rows, with shape
                ``[B, R_train]``. Long class indices for classification, or
                float values for regression.

        Returns:
            Tensor with shape ``[B, R, K * D]``, where ``K`` is the number of
            readout tokens and ``D`` is ``channels``.
        """
        B, R, C = x.size()
        R_train = y.size(-1)
        G, D = self.lin.in_features, self.lin.out_features
        K = self.readout_token.size(-2)

        # Feature grouping: gather G columns into each token.
        shift = 2 ** torch.arange(G, device=x.device)
        index = torch.arange(C, device=x.device)
        index = (index.view(C, 1) + shift.view(1, G)) % C  # [C, G]
        x = x[:, :, index]  # [B, R, C, G]
        x = self.lin(x)  # [B, R, C, D]

        if self.task_type == TaskType.classification:
            y_emb = self.y_emb(y).view(B, R_train, 1, D)
        else:
            y_emb = self.y_lin(y.unsqueeze(-1)).view(B, R_train, 1, D)

        x[:, :R_train] += y_emb.to(x.dtype)

        # Column-wise induced set attention (B * C as the batch axis): the
        # inducing points attend only over the first `R_train` (context) rows,
        # keeping target information from leaking into the row representations.
        x = x.transpose(1, 2)  # [B, C, R, D]
        for col_layer in self.col_layers:
            x = col_layer(
                query=x,  # [B, C, R, D]
                key_value=x[..., :R_train, :],  # [B, C, R_train, D]
            )  # [B, C, R, D]
        x = x.transpose(1, 2)  # [B, R, C, D]

        # Prepend readout tokens before row-wise attention.
        x = torch.cat(
            [self.readout_token.to(x.dtype).expand(B, R, K, D), x],
            dim=-2,
        )  # [B, R, K + C, D]

        # Row-wise attention (B * R as the batch axis).
        for i, row_layer in enumerate(self.row_layers):
            if i == len(self.row_layers) - 1:
                x = row_layer(  # Final layer extracts the readout tokens.
                    query=x[:, :, :K],  # [B, R, K, D]
                    key_value=x,  # [B, R, K + C, D]
                    rope=self.rope,
                )  # [B, R, K, D]
            else:
                x = row_layer(  # Intermediate full self-attention.
                    query=x,  # [B, R, K + C, D]
                    key_value=x,
                    rope=self.rope,
                )  # [B, R, K + C, D]

        x = self.norm(x)
        return x.view(B, R, K * D)  # [B, R, K * D]
