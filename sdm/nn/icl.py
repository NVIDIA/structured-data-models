"""In-context learning modules for structured tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, LayerNorm, Linear, ModuleList, Sequential

from sdm.nn.attention import TransformerBlock


class ICLBlock(torch.nn.Module):
    r"""Non-cached in-context learning block.

    The first ``y_train.size(0)`` rows of ``x`` are treated as labeled
    in-context examples. Remaining rows are prediction rows. Floating
    ``y_train`` selects regression output; integer labels select
    classification output.

    Args:
        channels: Input and hidden channel count.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of the transformer MLPs. If
            omitted, this defaults to ``2 * channels``.
        qassmax: Whether transformer attention uses QASSMax query scaling.
        norm_bias: Whether LayerNorm layers use a learnable bias.
        max_quantiles: Regression output width. These are raw head outputs;
            callers own sorting or task-specific postprocessing.
        max_classes: Classification output width and label vocabulary size.
        device: Parameter device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        channels: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        feedforward_channels: int | None = None,
        qassmax: bool = True,
        norm_bias: bool = True,
        max_quantiles: int = 999,
        max_classes: int = 10,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("`num_layers` must be at least 1")
        if max_quantiles < 1:
            raise ValueError("`max_quantiles` must be at least 1")
        if max_classes < 1:
            raise ValueError("`max_classes` must be at least 1")
        if feedforward_channels is None:
            feedforward_channels = 2 * channels
        if feedforward_channels < 1:
            raise ValueError("`feedforward_channels` must be at least 1")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.channels = channels
        self.max_quantiles = max_quantiles
        self.max_classes = max_classes

        self.y_cls_emb = Embedding(max_classes, channels, **factory_kwargs)
        self.y_reg_emb = Linear(1, channels, **factory_kwargs)

        self.layers = ModuleList(
            [
                TransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    qassmax=qassmax,
                    norm_bias=norm_bias,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.mlp = Sequential(
            LayerNorm(channels, **factory_kwargs),
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
        )
        self.cls_head = Linear(2 * channels, max_classes, **factory_kwargs)
        self.reg_head = Linear(2 * channels, max_quantiles, **factory_kwargs)

    def _validate_inputs(self, x: Tensor, y_train: Tensor) -> None:
        if x.dim() != 2:
            raise ValueError("`x` must have rank 2 with shape [num_rows, C]")
        if not x.is_floating_point():
            raise ValueError("`x` must be a floating-point tensor")
        if x.size(-1) != self.channels:
            raise ValueError(
                f"`x` must have {self.channels} channels, "
                f"got {x.size(-1)}"
            )
        if y_train.dim() != 1:
            raise ValueError(
                "`y_train` must have rank 1 with shape [num_train]"
            )
        if y_train.device != x.device:
            raise ValueError("`x` and `y_train` must be on the same device")
        if y_train.size(0) == 0:
            raise ValueError("`y_train` must contain at least one train row")
        if y_train.size(0) > x.size(0):
            raise ValueError(
                "`y_train` length must be less than or equal to "
                "`x.size(0)`"
            )
        if y_train.is_floating_point():
            return
        if y_train.is_complex():
            raise ValueError(
                "`y_train` must be floating point or integer labels"
            )
        labels = y_train.long()
        has_invalid_label = bool((labels < 0).any()) or bool(
            (labels >= self.max_classes).any()
        )
        if has_invalid_label:
            raise ValueError(
                "`y_train` classification labels must be in the range "
                f"[0, {self.max_classes})"
            )

    def forward(self, x: Tensor, y_train: Tensor) -> Tensor:
        r"""Run in-context prediction for the test rows of ``x``.

        Args:
            x: Row embeddings with shape ``[num_rows, channels]``.
            y_train: Train labels with shape ``[num_train]``.

        Returns:
            Classification logits or raw regression outputs for test rows only.
        """
        self._validate_inputs(x=x, y_train=y_train)

        num_train = y_train.size(0)
        if y_train.is_floating_point():
            y_train_emb = self.y_reg_emb(
                y_train.to(dtype=x.dtype).reshape(-1, 1)
            )
        else:
            y_train_emb = self.y_cls_emb(y_train.long())

        x = torch.cat(
            [
                x[:num_train] + y_train_emb,
                x[num_train:],
            ],
            dim=0,
        )

        if num_train == x.size(0):
            x = x[:0]
        else:
            for i, layer in enumerate(self.layers):
                query = x[num_train:] if i == len(self.layers) - 1 else x
                x = layer(
                    query=query.unsqueeze(0),
                    key_value=x[:num_train].unsqueeze(0),
                ).squeeze(0)

        out = self.mlp(x)
        if y_train.is_floating_point():
            return self.reg_head(out)
        return self.cls_head(out)
