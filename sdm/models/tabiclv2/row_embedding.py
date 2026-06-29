"""Row embedding module for structured tensor models."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList, Parameter

from sdm.nn import InducedTransformerBlock, RotaryEmbedding, TransformerBlock


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
        num_classes: The number of supported classes for classification.
            Set to ``0`` for regression.
        channels: The number of hidden channels.
        num_layers: The number of column/row attention layers.
        num_heads: The number of attention heads.
        group_size: The number of columns per feature group.
        num_inducing_points: The number of learnable inducing points.
        num_readout_tokens: The number of readout tokens.
        feedforward_channels: The hidden width of the MLP.
        qassmax: Whether to scale queries in the
            :class:`~sdm.nn.InducedTransformerBlock` with :class:`QASSMax`.
        norm_bias: Whether LayerNorm layers use a learnable bias.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        group_size: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        norm_bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.lin = Linear(group_size, channels, **factory_kwargs)

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, **factory_kwargs)

        self.col_layers = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=2 * channels,
                num_inducing_points=num_inducing_points,
                qassmax=True,
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
                feedforward_channels=2 * channels,
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
        train_mask: Tensor | None = None,  # [R],
    ) -> Tensor:  # [B, R, K * D]
        r"""The forward pass.

        Args:
            x: The feature tensor with shape ``[B, R, C]`` for ``B`` tables,
                ``R`` rows, and ``C`` columns.
            y: The targets with shape ``[B, R_train]``.
            train_mask: Training mask that denote the ``R_train`` rows along
                ``R`` in ``x`` that refer to the in-context examples.
                If not given, the first ``R_train`` rows along ``R`` refer to
                the in-context examples.

        Returns:
            Tensor with shape ``[B, R, K * D]``, where ``K`` is the number of
            readout tokens and ``D`` is ``channels``.
        """
        B, R, C = x.size()
        R_train = y.size(-1)
        G, D = self.lin.in_features, self.lin.out_features
        K = self.readout_token.size(-2)
        train_mask: Any = slice(R_train) if train_mask is None else train_mask

        # Feature grouping: gather G columns into each token.
        shift = 2 ** torch.arange(G, device=x.device)
        index = torch.arange(C, device=x.device)
        index = (index.view(C, 1) + shift.view(1, G)) % C  # [C, G]
        x = x[:, :, index]  # [B, R, C, G]
        x = self.lin(x)  # [B, R, C, D]

        if self.y_emb is not None:
            y_emb = self.y_emb(y).view(B, R_train, 1, D)
        else:
            assert self.y_lin is not None
            y_emb = self.y_lin(y.unsqueeze(-1)).view(B, R_train, 1, D)

        x[:, train_mask] += y_emb.to(x.dtype)

        # Column-wise induced set attention (B * C as the batch axis):
        x = x.transpose(1, 2)  # [B, C, R, D]
        for col_layer in self.col_layers:
            x = col_layer(
                query=x,  # [B, C, R, D]
                key_value=x[:, :, train_mask],  # [B, C, R_train, D]
            )  # [B, C, R, D]

        x = torch.cat(  # Prepend readout tokens before row-wise attention.
            [
                self.readout_token.to(x.dtype).expand(B, R, K, D),
                x.transpose(1, 2),  # [B, R, C, D]
            ],
            dim=-2,
        )  # [B, R, K + C, D]

        # Row-wise attention (B * R as the batch axis).
        for i, row_layer in enumerate(self.row_layers):
            x = row_layer(
                query=x[:, :, :K] if i == len(self.row_layers) - 1 else x,
                key_value=x,  # [B, R, K + C, D]
                rope=self.rope,
            )  # [B, R, K + C, D] or [B, R, K, D]

        return self.norm(x).view(B, R, K * D)  # [B, R, K * D]
