from typing import Any

import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.models.tabfm.icl import ICLBlock
from sdm.models.tabfm.row_embedding import RowEmbedding


class _TabFM(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int = 256,
        num_embedding_layers: int = 3,
        num_embedding_repeats: int = 2,
        num_embedding_col_heads: int = 4,
        num_embedding_row_heads: int = 8,
        num_inducing_points: int = 256,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 8,
        num_icl_layers: int = 24,
        num_icl_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.cell_embedding = CellEmbedding(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )
        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_embedding_layers,
            num_repeats=num_embedding_repeats,
            num_col_heads=num_embedding_col_heads,
            num_row_heads=num_embedding_row_heads,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=max(1, num_classes),
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, 1 or num_classes]
        x = self.cell_embedding(x, categorical_mask)
        x = self.row_embedding(x, y, cache=cache)
        return self.icl_block(x, y, cache=cache)
