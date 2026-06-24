"""In-context learning modules for structured tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    GELU,
    Embedding,
    LayerNorm,
    Linear,
    ModuleList,
    Sequential,
)

from sdm.nn.attention import TransformerBlock
from sdm.task import TaskType


class ICLBlock(torch.nn.Module):
    r"""In-context learning block for classification and regression.

    Training-row label embeddings are injected into the row hidden states,
    after which a stack of cross-attention :class:`TransformerBlock` layers
    lets every row attend over the in-context training rows. A shared MLP
    followed by a task-specific linear head maps the test-row states to
    predictions: class logits for classification, or quantile predictions for
    regression.

    The design follows the in-context learning setup of the `"TabICLv2: A
    better, faster, scalable, and open tabular foundation model"
    <https://arxiv.org/abs/2602.11139>`_ paper.

    Args:
        task_type: A :class:`~sdm.TaskType` selecting the prediction head.
        channels: The number of row hidden channels.
        num_layers: The number of cross-attention transformer layers.
        num_heads: The number of attention heads per layer.
        num_classes: The number of target classes for classification, i.e. the
            size of the label embedding table and the classification head.
            Unused for regression.
        num_quantiles: The number of predicted quantiles for regression, i.e.
            the size of the regression head. Unused for classification.
        norm_bias: Whether :class:`~torch.nn.LayerNorm` layers use a learnable
            bias.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.
    """

    def __init__(
        self,
        task_type: TaskType,
        channels: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_classes: int = 10,
        num_quantiles: int = 999,
        norm_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.task_type = task_type

        if task_type == TaskType.classification:
            self.y_emb: torch.nn.Module = Embedding(
                num_embeddings=num_classes,
                embedding_dim=channels,
                **factory_kwargs,
            )
        else:
            assert task_type == TaskType.regression
            self.y_lin = Linear(1, channels, bias=False, **factory_kwargs)

        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=2 * channels,
                qassmax=True,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            self.layers.append(layer)

        self.mlp = Sequential(
            LayerNorm(channels, bias=norm_bias, **factory_kwargs),
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
        )
        if task_type == TaskType.classification:
            self.head = Linear(2 * channels, num_classes, **factory_kwargs)
        else:
            self.head = Linear(2 * channels, num_quantiles, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [B, R, C]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R_test, *]
        r"""Forward pass of :class:`ICLBlock`.

        Args:
            x: Row hidden states with shape ``[B, R, C]``. The first
                ``R_train`` rows along ``R`` are the in-context training rows.
            y: Training labels with shape ``[B, R_train]``.

        Returns:
            For classification, class logits with shape
            ``[B, R_test, num_classes]``. For regression, quantile predictions
            with shape ``[B, R_test, num_quantiles]``.
        """
        R_train = y.size(-1)

        if self.task_type == TaskType.classification:
            y_emb = self.y_emb(y)  # [B, R_train, C]
        else:
            assert self.task_type == TaskType.regression
            y_emb = self.y_lin(y.unsqueeze(-1))  # [B, R_train, C]

        x[:, :R_train] += y_emb.to(x.dtype)

        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            x = layer(
                query=x[:, R_train:] if is_last else x,  # last: test rows only
                key_value=x[:, :R_train],
            )

        out = self.mlp(x)  # [B, R_test, 2 * C]
        return self.head(out)  # [B, R_test, num_classes or num_quantiles]
