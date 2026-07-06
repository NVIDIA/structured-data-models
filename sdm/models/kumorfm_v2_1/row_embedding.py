"""Row embedding for KumoRFM v2.1."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList, Parameter

from sdm.nn import InducedTransformerBlock, RotaryEmbedding, TransformerBlock


class RowEmbedding(torch.nn.Module):
    r"""Embed processed rows for KumoRFM v2.1.

    Args:
        channels: Hidden channel count per feature token.
        num_layers: Number of column and row attention layers.
        num_heads: Number of attention heads.
        group_size: Number of shifted features grouped into each token.
        num_inducing_points: Number of inducing points in column attention.
        num_readout_tokens: Number of row readout tokens.
        max_classes: Maximum number of classification labels.
        device: Parameter device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        channels: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        group_size: int = 3,
        num_inducing_points: int = 128,
        num_readout_tokens: int = 4,
        max_classes: int = 10,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if max_classes < 1:
            raise ValueError("`max_classes` must be at least 1")
        if num_layers < 1:
            raise ValueError("`num_layers` must be at least 1")
        if group_size < 1:
            raise ValueError("`group_size` must be at least 1")
        if num_readout_tokens < 1:
            raise ValueError("`num_readout_tokens` must be at least 1")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.max_classes = max_classes

        self.lin = Linear(group_size, channels, **factory_kwargs)
        self.y_cls_lin = Embedding(max_classes, channels, **factory_kwargs)
        self.y_reg_lin = Linear(1, channels, **factory_kwargs)

        self.col_layers = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=2 * channels,
                num_inducing_points=num_inducing_points,
                qassmax=True,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.readout_token = Parameter(
            torch.empty((1, num_readout_tokens, channels), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        self.row_layers = ModuleList(
            TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=2 * channels,
                qassmax=False,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.rope = RotaryEmbedding(
            channels=channels // num_heads,
            theta=100_000,
            **factory_kwargs,
        )
        self.norm = LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [R, C]
        y_train: Tensor,  # [R_train]
        train_mask: Tensor | None = None,  # [R]
        max_train: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:  # [R, K * D]
        r"""Embed rows using selected labeled rows as column context.

        Floating-point targets use the regression encoder. Integer targets
        use the classification encoder. When ``train_mask`` is omitted,
        targets correspond to a prefix of rows. An empty context falls back
        to all rows in the local input.

        Args:
            x: Processed floating-point features with shape ``[R, C]``.
            y_train: Targets corresponding to selected rows, with shape
                ``[R_train]``.
            train_mask: Optional boolean mask selecting target rows.
            max_train: Optional maximum context rows sampled independently at
                each column layer.
            generator: Optional generator for per-layer context subsampling.

        Returns:
            Normalized row embeddings with shape ``[R, K * D]``.
        """
        self._validate_inputs(
            x=x,
            y_train=y_train,
            train_mask=train_mask,
            max_train=max_train,
        )

        R, C = x.size()
        G, D = self.lin.in_features, self.lin.out_features
        K = self.readout_token.size(1)
        if train_mask is None:
            train_index = torch.arange(y_train.numel(), device=x.device)
        else:
            train_index = train_mask.nonzero(as_tuple=True)[0]
        context_index = (
            train_index
            if train_index.numel() > 0
            else torch.arange(R, device=x.device)
        )

        # Gather shifted groups of G features into each feature token.
        shift = 2 ** torch.arange(G, device=x.device)
        index = torch.arange(C, device=x.device)
        index = (index.view(C, 1) + shift.view(1, G)) % C  # [C, G]
        x = self.lin(x[:, index])  # [R, C, D]

        if y_train.numel() > 0:
            if y_train.is_floating_point():
                y_emb = self.y_reg_lin(y_train.to(dtype=x.dtype).unsqueeze(-1))
            else:
                y_emb = self.y_cls_lin(y_train.long())
            label_delta = x.new_zeros((R, D)).index_add(
                0,
                train_index,
                y_emb.to(x.dtype),
            )
            x = x + label_delta.unsqueeze(-2)

        # Column attention treats features as the batch axis.
        x = x.transpose(0, 1)  # [C, R, D]
        for col_layer in self.col_layers:
            layer_context = self._subsample_context(
                context_index,
                max_train=max_train,
                generator=generator,
            )
            x = col_layer(
                query=x,
                key_value=x.index_select(-2, layer_context),
            )  # [C, R, D]

        x = torch.cat(
            [
                self.readout_token.to(x.dtype).expand(R, K, D),
                x.transpose(0, 1),  # [R, C, D]
            ],
            dim=-2,
        )  # [R, K + C, D]

        # Row attention treats rows as the batch axis.
        for i, row_layer in enumerate(self.row_layers):
            x = row_layer(
                query=x[:, :K] if i == len(self.row_layers) - 1 else x,
                key_value=x,
                rope=self.rope,
            )  # [R, K + C, D] or [R, K, D]

        return self.norm(x).view(R, K * D)

    def _subsample_context(
        self,
        context_index: Tensor,
        *,
        max_train: int | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if max_train is None or context_index.numel() <= max_train:
            return context_index

        sample_device = (
            context_index.device if generator is None else generator.device
        )
        permutation = torch.randperm(
            context_index.numel(),
            device=sample_device,
            generator=generator,
        )[:max_train]
        return context_index.index_select(
            0, permutation.to(context_index.device)
        )

    def _validate_inputs(
        self,
        *,
        x: Tensor,
        y_train: Tensor,
        train_mask: Tensor | None,
        max_train: int | None,
    ) -> None:
        if x.dim() != 2:
            raise ValueError("`x` must have shape [num_rows, num_columns]")
        if not x.is_floating_point():
            raise TypeError("`x` must be a floating-point tensor")
        if x.size(0) == 0:
            raise ValueError("`x` must contain at least one row")
        if x.size(1) == 0:
            raise ValueError("`x` must contain at least one column")
        if y_train.dim() != 1:
            raise ValueError("`y_train` must have shape [num_train]")
        if y_train.device != x.device:
            raise ValueError("`x` and `y_train` must be on the same device")
        if y_train.is_complex():
            raise TypeError(
                "`y_train` must contain floating-point or integer labels"
            )
        if max_train is not None and (
            isinstance(max_train, bool) or not isinstance(max_train, int)
        ):
            raise TypeError("`max_train` must be an integer or None")
        if max_train is not None and max_train < 1:
            raise ValueError("`max_train` must be at least 1")

        if train_mask is None:
            if y_train.numel() > x.size(0):
                raise ValueError(
                    "`y_train` cannot contain more labels than rows"
                )
        else:
            if train_mask.dim() != 1:
                raise ValueError("`train_mask` must have shape [num_rows]")
            if train_mask.size(0) != x.size(0):
                raise ValueError(
                    "`train_mask` length must match the rows in `x`"
                )
            if train_mask.dtype != torch.bool:
                raise TypeError("`train_mask` must have boolean dtype")
            if train_mask.device != x.device:
                raise ValueError(
                    "`train_mask` and `x` must be on the same device"
                )
            if int(train_mask.count_nonzero()) != y_train.numel():
                raise ValueError(
                    "`y_train` label count must match rows selected by "
                    "`train_mask`"
                )

        if not y_train.is_floating_point() and y_train.numel() > 0:
            invalid = (y_train < 0) | (y_train >= self.max_classes)
            if bool(invalid.any()):
                raise ValueError(
                    f"Classification labels must be in [0, {self.max_classes})"
                )
