from typing import Literal

import torch
from torch import Tensor
from torch.nn import Linear

from sdm.models.tabfm.cell_embedding import CellEmbedding


class FourierNanIndicatorCellEmbedding(CellEmbedding):
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
                ``[..., 1, C]`` for query-only cached inference.
            batch_size_limit: Target maximum number of cells per Fourier
                feature chunk.
            out: Optional preallocated output buffer.

        Returns:
            Cell embeddings with shape ``[..., R, C, D]``.
        """
        missing = x.isnan()

        if impute_mean is None:
            if train_size is None:
                raise ValueError(
                    "FourierNanIndicatorCellEmbedding requires train_size "
                    "or impute_mean"
                )
            # Impute from observed context rows only, with fp32 statistics.
            train = x[..., :train_size, :].to(torch.float32)
            observed = ~missing[..., :train_size, :]
            count = observed.sum(dim=-2, keepdim=True).clamp(min=1)
            mean = train.masked_fill(~observed, 0.0).sum(dim=-2, keepdim=True)
            impute_mean = (mean / count).to(x.dtype)

        x = torch.where(missing, impute_mean.to(x.dtype), x)

        result = super().forward(
            x,
            categorical_mask,
            batch_size_limit=batch_size_limit,
            out=out,
        )

        C = x.size(-1)
        index = self._group_index(C, x.device)
        grouped_missing = missing[..., index]  # [..., R, C, G]
        missing_embedding = self.nan_lin(
            grouped_missing.to(self.nan_lin.weight.dtype)
        )
        if out is None:
            return result + missing_embedding
        result.add_(missing_embedding.to(result.dtype))
        return result
