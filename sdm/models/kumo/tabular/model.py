from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from sdm.cache import Cache
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.models.kumo.tabular.table_encoder import TableEncoder
from sdm.models.tabfm.cell_embedding import CellEmbedding


class _KumoTabular(torch.nn.Module):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.cell_embedding = CellEmbedding(
            channels=128,
            group_size=3,
            num_frequencies=32,
            **factory_kwargs,
        )
        self.y_encoder = Linear(10, 128, **factory_kwargs)
        self.table_encoder = TableEncoder(**factory_kwargs)
        self.icl_block = ICLBlock(
            num_classes=10,
            out_channels=10,
            channels=512,
            num_layers=12,
            num_heads=8,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_context]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_query, 10]
        num_context_rows = y.size(-1)
        x = self.cell_embedding(x, categorical_mask)

        y_one_hot = F.one_hot(y.long(), num_classes=10)
        y_embedding = self.y_encoder(
            y_one_hot.to(self.y_encoder.weight.dtype)
        ).to(x.dtype)
        x = torch.cat(
            (
                x[..., :num_context_rows, :, :] + y_embedding.unsqueeze(-2),
                x[..., num_context_rows:, :, :],
            ),
            dim=-3,
        )

        x = self.table_encoder(x, num_context_rows, cache=cache)
        return self.icl_block(
            x,
            y,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
