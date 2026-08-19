from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList, Parameter, RMSNorm, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import InducedTransformerBlock, RotaryEmbedding


class TableEncoder(torch.nn.Module):
    """Encode table cells into row representations for Kumo Tabular.

    Args:
        channels: Number of channels per cell token.
        num_col_heads: Number of column-attention heads.
        num_row_heads: Number of row-attention heads.
        num_inducing_points: Number of inducing points in each column block.
        num_cls_tokens: Number of classification tokens per row.
        num_stages: Number of alternating column and row stages.
        device: Device of the parameters.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        channels: int = 128,
        num_col_heads: int = 8,
        num_row_heads: int = 8,
        num_inducing_points: int = 128,
        num_cls_tokens: int = 4,
        num_stages: int = 4,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError("'num_stages' must be at least 1")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = Parameter(
            torch.empty(num_cls_tokens, channels, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        rope = RotaryEmbedding(
            channels=channels // num_row_heads,
            layout="split_half",
            theta=100_000,
            requires_grad=False,
            partial_rotary_factor=0.25,
            **factory_kwargs,
        )

        self.col_blocks = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_inducing_points=num_inducing_points,
                inducing_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_col_heads,
                    **factory_kwargs,
                ),
                output_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_col_heads,
                    **factory_kwargs,
                ),
                **factory_kwargs,
            )
            for _ in range(num_stages)
        )
        self.col_projections = ModuleList(
            Sequential(
                Linear(channels, channels, **factory_kwargs),
                RMSNorm(channels, **factory_kwargs),
            )
            for _ in range(num_stages)
        )
        self.row_blocks = ModuleList(
            KumoTabularTransformerBlock(
                channels=channels,
                num_heads=num_row_heads,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_stages)
        )
        self.row_norms = ModuleList(
            RMSNorm(channels, **factory_kwargs) for _ in range(num_stages)
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C, D]
        num_context_rows: int,
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R, K * D]
        """Encode table cells into row representations.

        Args:
            x: Cell tokens with shape ``[..., R, C, D]``, where ``R`` is the
                number of rows, ``C`` is the number of columns, and ``D`` is
                the number of channels.
            num_context_rows: Number of context rows at the start of ``R``.
            cache: Optional key/value cache for context reuse.

        Returns:
            Row representations with shape ``[..., R, K * D]``, where ``K``
            is the number of classification tokens.
        """
        *batch, num_rows, _, channels = x.size()
        num_cls_tokens = self.num_cls_tokens

        for i, (col_block, col_projection, row_block, row_norm) in enumerate(
            zip(
                self.col_blocks,
                self.col_projections,
                self.row_blocks,
                self.row_norms,
            )
        ):
            # CLS tokens bypass column stages after their first insertion.
            features = x if i == 0 else x[..., num_cls_tokens:, :]
            features = features.transpose(-2, -3).contiguous()
            key = f"table_encoder.col_block{i}"
            if cache is not None and cache.is_replaying:
                key_value = cast(KVCacheEntry, cache[key])
            else:
                key_value = features[..., :num_context_rows, :]

            result = col_block(
                query=features,
                key_value=key_value,
                return_key_value=cache is not None and cache.is_recording,
            )
            if cache is not None and cache.is_recording:
                features, cache[key] = result
            else:
                features = result
            features = col_projection(features.transpose(-2, -3))

            if i == 0:
                cls_tokens = self.cls_tokens.to(x.dtype)
                cls_tokens = cls_tokens.view(
                    *(1,) * len(batch), 1, num_cls_tokens, channels
                ).expand(*batch, num_rows, -1, -1)
                x = torch.cat((cls_tokens, features), dim=-2)
            else:
                x = torch.cat((x[..., :num_cls_tokens, :], features), dim=-2)

            query = x
            if i == len(self.row_blocks) - 1:
                query = x[..., :num_cls_tokens, :]
            x = row_norm(row_block(query=query, key_value=x))

        return x.flatten(-2)
