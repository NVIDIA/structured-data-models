"""Dataset-level in-context prediction for KumoRFM v2.1."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, LayerNorm, Linear, ModuleList, Sequential

from sdm.nn import TransformerBlock


class ICLPredictor(torch.nn.Module):
    r"""Predict targets from labeled context rows.

    Every transformer layer attends only to context rows, and the final layer
    emits states only for the remaining test rows. Floating-point targets
    select regression; boolean and integer targets select classification.

    Args:
        channels: Input and hidden channel count.
        num_layers: Number of dataset-level transformer layers.
        num_heads: Number of attention heads per transformer layer.
        num_quantiles: Number of regression quantile values to predict.
        max_classes: Classification output width and label vocabulary size.
        device: Parameter device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        channels: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_quantiles: int = 999,
        max_classes: int = 10,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("channels", channels),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("num_quantiles", num_quantiles),
            ("max_classes", max_classes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer")
            if value < 1:
                raise ValueError(f"`{name}` must be at least 1")
        if channels % num_heads != 0:
            raise ValueError(
                f"`channels` ({channels}) must be divisible by "
                f"`num_heads` ({num_heads})"
            )

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.channels = channels
        self.num_quantiles = num_quantiles
        self.max_classes = max_classes

        self.y_cls_emb = Embedding(max_classes, channels, **factory_kwargs)
        self.y_reg_lin = Linear(1, channels, **factory_kwargs)

        self.layers = ModuleList(
            TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=2 * channels,
                qassmax=True,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.prediction_mlp = Sequential(
            LayerNorm(channels, **factory_kwargs),
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
        )
        self.cls_head = Linear(
            2 * channels,
            max_classes,
            **factory_kwargs,
        )
        self.reg_head = Linear(
            2 * channels,
            num_quantiles,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [R, D]
        y_train: Tensor,  # [T]
        context_mask: Tensor | None = None,  # [R]
    ) -> Tensor:  # [R - T, max_classes or num_quantiles]
        r"""Predict targets for rows outside the labeled context.

        Args:
            x: Row embeddings with shape ``[R, D]``.
            y_train: Context targets with shape ``[T]``.
            context_mask: Optional boolean mask selecting context rows. When
                omitted, the first ``T`` rows are used.

        Returns:
            Classification logits with shape ``[R - T, max_classes]`` for
            boolean or integer targets, or regression quantile values with
            shape ``[R - T, num_quantiles]`` for floating-point targets. The
            regression values are raw, unsorted quantile-head outputs.
        """
        self._validate_inputs(
            x=x,
            y_train=y_train,
            context_mask=context_mask,
        )

        num_train = y_train.size(0)
        if context_mask is None:
            context_index = torch.arange(num_train, device=x.device)
            test_index = torch.arange(num_train, x.size(0), device=x.device)
        else:
            context_index = context_mask.nonzero(as_tuple=True)[0]
            test_index = (~context_mask).nonzero(as_tuple=True)[0]

        is_regression = y_train.is_floating_point()
        if is_regression:
            y_emb = self.y_reg_lin(y_train.to(dtype=x.dtype).unsqueeze(-1))
        else:
            y_emb = self.y_cls_emb(y_train.long())

        if test_index.numel() == 0:
            output_size = (
                self.num_quantiles if is_regression else self.max_classes
            )
            return x.new_empty((0, output_size))

        x = torch.cat(
            [
                x.index_select(0, context_index) + y_emb,
                x.index_select(0, test_index),
            ],
            dim=0,
        )
        for i, layer in enumerate(self.layers):
            x = layer(
                query=x[num_train:] if i == len(self.layers) - 1 else x,
                key_value=x[:num_train],
            )

        x = self.prediction_mlp(x)
        if is_regression:
            return self.reg_head(x)
        return self.cls_head(x)

    def _validate_inputs(
        self,
        *,
        x: Tensor,
        y_train: Tensor,
        context_mask: Tensor | None,
    ) -> None:
        if x.dim() != 2:
            raise ValueError("`x` must have shape [num_rows, channels]")
        if x.size(1) != self.channels:
            raise ValueError(
                f"`x` must have {self.channels} channels, got {x.size(1)}"
            )
        if not x.is_floating_point():
            raise TypeError("`x` must be a floating-point tensor")
        if y_train.dim() != 1:
            raise ValueError("`y_train` must have shape [num_train]")
        if y_train.is_complex():
            raise TypeError(
                "`y_train` must contain floating-point, boolean, or integer "
                "targets"
            )

        parameter = self.y_reg_lin.weight
        if x.device != parameter.device:
            raise ValueError("`x` must be on the same device as the module")
        if y_train.device != x.device:
            raise ValueError("`x` and `y_train` must be on the same device")

        num_train = y_train.size(0)
        if num_train == 0:
            raise ValueError("`y_train` must contain at least one context row")
        if context_mask is None:
            if num_train > x.size(0):
                raise ValueError(
                    "`y_train` cannot contain more targets than rows in `x`"
                )
        else:
            if context_mask.dim() != 1 or context_mask.size(0) != x.size(0):
                raise ValueError("`context_mask` must have shape [num_rows]")
            if context_mask.dtype != torch.bool:
                raise TypeError("`context_mask` must have boolean dtype")
            if context_mask.device != x.device:
                raise ValueError(
                    "`context_mask` and `x` must be on the same device"
                )
            if int(context_mask.count_nonzero()) != num_train:
                raise ValueError(
                    "`y_train` target count must match rows selected by "
                    "`context_mask`"
                )

        if not y_train.is_floating_point():
            invalid = (y_train < 0) | (y_train >= self.max_classes)
            if bool(invalid.any()):
                raise ValueError(
                    f"Classification labels must be in [0, {self.max_classes})"
                )
