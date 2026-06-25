"""TabICLv2 tabular foundation model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sdm.model.tabiclv2.icl import ICLBlock
from sdm.model.tabiclv2.row_embedding import RowEmbedding
from sdm.task import TaskType


class TabICLv2(torch.nn.Module):
    r"""The TabICLv2 tabular foundation model from the TabICLv2 paper.

    Introduced in `"TabICLv2: A better, faster, scalable, and open tabular
    foundation model" <https://arxiv.org/abs/2602.11139>`_, the model first
    encodes a table into per-row embeddings with
    :class:`~sdm.model.tabiclv2.RowEmbedding`, then makes in-context
    predictions for the test rows with
    :class:`~sdm.model.tabiclv2.ICLBlock`.

    The :class:`~sdm.model.tabiclv2.ICLBlock` operates on the concatenated
    readout tokens produced by :class:`~sdm.model.tabiclv2.RowEmbedding`, so
    its channel count is derived as ``num_readout_tokens * channels``.

    Args:
        task_type: A :class:`~sdm.TaskType` selecting the prediction head.
        channels: The per-token channel count of the row encoder.
        num_embedding_layers: The number of row-encoder attention layers.
        num_icl_layers: The number of in-context cross-attention layers.
        num_heads: The number of attention heads in both stages.
        group_size: The number of columns grouped into each encoder token.
        num_inducing_points: The number of inducing points in the row encoder.
        num_readout_tokens: The number of readout tokens produced per row.
        norm_bias: Whether :class:`~torch.nn.LayerNorm` layers use a learnable
            bias.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.
    """

    def __init__(
        self,
        task_type: TaskType,
        channels: int = 128,
        num_embedding_layers: int = 3,
        num_icl_layers: int = 12,
        num_heads: int = 8,
        group_size: int = 3,
        num_inducing_points: int = 128,
        num_readout_tokens: int = 4,
        norm_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.task_type = task_type

        self.row_embedding = RowEmbedding(
            task_type=task_type,
            channels=channels,
            num_layers=num_embedding_layers,
            num_heads=num_heads,
            group_size=group_size,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.icl = ICLBlock(
            task_type=task_type,
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_heads,
            norm_bias=norm_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [B, R, C]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R_test, *]
        r"""Forward pass of :class:`TabICLv2`.

        Args:
            x: Feature tensor with shape ``[B, R, C]`` for ``B`` tables, ``R``
                rows, and ``C`` columns. The first ``R_train`` rows are the
                in-context training rows and the rest are the test rows.
            y: Training targets with shape ``[B, R_train]``. Long class indices
                for classification, or float values for regression.

        Returns:
            For classification, class logits with shape ``[B, R_test, 10]``.
            For regression, quantile predictions with shape
            ``[B, R_test, 999]``.
        """
        x = self.row_embedding(x, y)
        return self.icl(x, y)
