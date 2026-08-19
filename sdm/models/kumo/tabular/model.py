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
        num_classes: int,
        num_quantiles: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_classes = num_classes

        self.cell_embedding = CellEmbedding(
            channels=128,
            group_size=3,
            num_frequencies=32,
            **factory_kwargs,
        )
        self.y_encoder = Linear(
            num_classes or 1,
            128,
            bias=num_classes > 0,
            **factory_kwargs,
        )
        self.table_encoder = TableEncoder(**factory_kwargs)
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=512,
            num_layers=12,
            num_heads=8,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        R_train = y.size(-1)
        x = self.cell_embedding(x, categorical_mask)

        if y.numel() > 0:
            if self.num_classes > 0:
                y_emb = F.one_hot(y.long(), num_classes=self.num_classes)
            else:
                y_emb = y.unsqueeze(-1)
            y_emb = self.y_encoder(y_emb.to(self.y_encoder.weight.dtype)).to(
                x.dtype
            )
            x = torch.cat(
                (
                    x[..., :R_train, :, :] + y_emb.unsqueeze(-2),
                    x[..., R_train:, :, :],
                ),
                dim=-3,
            )

        x = self.table_encoder(x, R_train, cache=cache)
        return self.icl_block(
            x=x,
            y=y,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
