# ruff: noqa: D205
"""TabICLv2 tabular foundation model."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding


class TabICLv2(torch.nn.Module):
    r"""The tabular foundation model from the `"TabICLv2: A better, faster,
    scalable, and open tabular foundation model"
    <https://arxiv.org/abs/2602.11139>`_ paper.

    Args:
        device: The device.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        super().__init__()

        self.cls_model = _TabICLv2(
            num_classes=10,
            num_quantiles=0,
            norm_bias=True,
            device=device,
        )
        self.reg_model = _TabICLv2(
            num_classes=0,
            num_quantiles=999,
            norm_bias=False,
            device=device,
        )

    def forward(  # TODO Add multi-class support.
        self,
        x: Tensor,  # [B, R, C]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R_test, num_classes or 999]
        r"""The forward pass.

        Args:
            x: The feature tensor with shape ``[B, R, C]`` for ``B`` tables,
                ``R`` rows, and ``C`` columns.
                The first ``R_train`` rows along ``R`` refer to the in-context
                examples.
            y: The targets with shape ``[B, R_train]``.

        Returns:
            Tensor with shape ``[B, R_test, num_classes]`` for integer ``y``
            and ``[B, R_test, 999]`` for floating-point ``y``.
            Integer ``y`` return class logits.
            Floating-point ``y`` return 999 quantiles at probability levels
            :math:`\left\{0.001, 0.002, \ldots, 0.999\right\}`.
        """
        if y.is_floating_point():
            return self.reg_model(x, y)
        return self.cls_model(x, y)


class _TabICLv2(torch.nn.Module):
    r"""The tabular foundation model from the TabICLv2 paper.

    Introduced in `"TabICLv2: A better, faster, scalable, and open tabular
    foundation model" <https://arxiv.org/abs/2602.11139>`_, the model first
    encodes a table into per-row embeddings with
    :class:`~sdm.models.tabiclv2.RowEmbedding`, then makes in-context
    predictions for the test rows with
    :class:`~sdm.models.tabiclv2.ICLBlock`.

    The :class:`~sdm.models.tabiclv2.ICLBlock` operates on the concatenated
    readout tokens produced by :class:`~sdm.models.tabiclv2.RowEmbedding`, so
    its channel count is derived as ``num_readout_tokens * channels``.

    Args:
        channels: The per-token channel count of the row encoder.
        num_embedding_layers: The number of row-encoder attention layers.
        num_icl_layers: The number of in-context cross-attention layers.
        num_heads: The number of attention heads in both stages.
        group_size: The number of columns grouped into each encoder token.
        num_inducing_points: The number of inducing points in the row encoder.
        num_readout_tokens: The number of readout tokens produced per row.
        norm_bias: Whether :class:`~torch.nn.LayerNorm` layers use a learnable
            bias.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        channels: int = 128,
        num_embedding_layers: int = 3,
        num_embedding_heads: int = 8,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_readout_tokens: int = 4,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        norm_bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            group_size=group_size,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.head = Sequential(
            Linear(
                in_features=num_readout_tokens * channels,
                out_features=2 * num_readout_tokens * channels,
                **factory_kwargs,
            ),
            GELU(),
            Linear(
                in_features=2 * num_readout_tokens * channels,
                out_features=num_classes or num_quantiles,
                **factory_kwargs,
            ),
        )

    def forward(
        self,
        x: Tensor,  # [B, R, C]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R_test, num_classes or num_quantiles]
        r"""The forward pass.

        Args:
            x: The feature tensor with shape ``[B, R, C]`` for ``B`` tables,
                ``R`` rows, and ``C`` columns.
                The first ``R_train`` rows along ``R`` refer to the in-context
                examples.
            y: The targets with shape ``[B, R_train]``.

        Returns:
            Tensor with shape ``[B, R_test, num_classes or num_quantiles]``.
        """
        x = self.row_embedding(x, y)
        x = self.icl_block(x, y)
        return self.head(x)
