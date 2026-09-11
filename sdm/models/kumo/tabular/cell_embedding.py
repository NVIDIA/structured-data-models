import math
from typing import Literal, cast

import torch
from torch import Tensor
from torch.nn import Linear

from sdm.cache import Cache
from sdm.models.tabfm.cell_embedding import CellEmbedding as TabFMCellEmbedding


class CellEmbedding(TabFMCellEmbedding):
    """Embed Fourier features with an additive NaN indicator.

    NaNs are imputed with the observed context-row mean of their column, or
    zero when the context has no observed value. The original NaN mask is
    grouped like the values and projected without bias, so finite inputs
    follow the unchanged Fourier path. Numerical and categorical cells share
    the NaN projection.

    Args:
        channels: Number of output channels per cell.
        group_size: Number of circularly grouped columns per cell, using
            offsets ``2**i - 1``.
        num_frequencies: Number of learned Fourier frequencies per group.
        device: Device on which to initialize the module.
        dtype: Data type in which to initialize the module.
    """

    def __init__(
        self,
        channels: int,
        group_size: int,
        num_frequencies: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            device=device,
            dtype=dtype,
        )
        self.nan_lin = Linear(
            group_size,
            channels,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        categorical_mask: Tensor,  # [..., C]
        *,
        train_size: int | None = None,
        impute_mean: Tensor | None = None,
        cache: Cache | None = None,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> Tensor:  # [..., R, C, D]
        """Embed cells using context-only NaN imputation.

        Args:
            x: Input values with shape ``[..., R, C]``.
            categorical_mask: Categorical column mask with shape ``[..., C]``.
            train_size: Number of leading context rows used for imputation
                when ``impute_mean`` is omitted.
            impute_mean: Optional precomputed column means with shape
                ``[..., 1, C]``.
            cache: Records imputation means for reuse in query-only calls.
                Replayed means take precedence over ``train_size`` and
                ``impute_mean``.
            batch_size_limit: Target maximum number of cells per Fourier
                feature chunk.
            out: Optional preallocated output buffer.

        Returns:
            Cell embeddings with shape ``[..., R, C, D]``.
        """
        missing = x.isnan()

        if cache is not None and cache.is_replaying:
            impute_mean = cast(Tensor, cache["cell_impute_mean"])
        elif impute_mean is None:
            if train_size is None:
                raise ValueError(
                    "CellEmbedding requires train_size or impute_mean"
                )
            # Impute from observed context rows only, with fp32 statistics.
            mean = x[..., :train_size, :].nanmean(
                dim=-2,
                keepdim=True,
                dtype=torch.float32,
            )
            impute_mean = mean.masked_fill(mean.isnan(), 0.0).to(x.dtype)

        if cache is not None and cache.is_recording:
            cache["cell_impute_mean"] = impute_mean

        x = torch.where(missing, impute_mean.to(x.dtype), x)

        result = super().forward(
            x,
            categorical_mask,
            batch_size_limit=batch_size_limit,
            out=out,
        )

        C, G = x.size(-1), self.group_size
        index = self._group_index(C, x.device)
        grouped_missing = missing.index_select(-1, index.view(-1))
        grouped_missing = grouped_missing.unflatten(-1, (C, G))
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            projected = self.nan_lin(
                grouped_missing.to(self.nan_lin.weight.dtype)
            ).to(result.dtype)
            return result.add_(projected)

        batch_size_limit = self._resolve_batch_size_limit(
            x=x,
            batch_size_limit=batch_size_limit,
        )
        rows_per_chunk = max(1, x.size(-2))
        if batch_size_limit is not None:
            cells_per_row = max(1, math.prod(x.shape[:-2]) * C)
            rows_per_chunk = max(1, batch_size_limit // cells_per_row)
        # Match the separate projection/add rounding in the differentiable
        # path, while bounding the temporary projection during inference.
        for start in range(0, x.size(-2), rows_per_chunk):
            grouped = grouped_missing[
                ..., start : start + rows_per_chunk, :, :
            ]
            projected = self.nan_lin(grouped.to(self.nan_lin.weight.dtype))
            result[..., start : start + rows_per_chunk, :, :].add_(
                projected.to(result.dtype)
            )
        return result
