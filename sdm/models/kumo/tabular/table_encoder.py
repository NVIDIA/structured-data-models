# ruff: noqa: D101, D102

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import ModuleList, Parameter, RMSNorm

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import InducedTransformerBlock, RotaryEmbedding


class TableEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_col_heads: int,
        num_row_heads: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_readout_tokens = num_readout_tokens
        self.cls_tokens = Parameter(
            torch.empty(num_readout_tokens, channels, **factory_kwargs)
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
                    query_log_scale=True,
                    **factory_kwargs,
                ),
                output_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_col_heads,
                    query_log_scale=False,
                    **factory_kwargs,
                ),
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.row_blocks = ModuleList(
            KumoTabularTransformerBlock(
                channels=channels,
                num_heads=num_row_heads,
                query_log_scale=False,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, C, D]
        num_context_rows: int,
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R, K * D]
        *batch, num_rows, _, channels = x.size()
        num_readout_tokens = self.num_readout_tokens

        for i, (col_block, row_block) in enumerate(
            zip(self.col_blocks, self.row_blocks, strict=True)
        ):
            # CLS readout tokens bypass column layers after their insertion.
            features = x if i == 0 else x[..., num_readout_tokens:, :]
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
            features = features.transpose(-2, -3)

            if i == 0:
                readout_tokens = self.cls_tokens.to(x.dtype)
                readout_tokens = readout_tokens.view(
                    *(1,) * len(batch), 1, num_readout_tokens, channels
                ).expand(*batch, num_rows, -1, -1)
                x = torch.cat((readout_tokens, features), dim=-2)
            else:
                x = torch.cat(
                    (x[..., :num_readout_tokens, :], features),
                    dim=-2,
                )

            query = x
            if i == len(self.row_blocks) - 1:
                query = x[..., :num_readout_tokens, :]
            x = row_block(query=query, key_value=x)

        return self.norm(x).flatten(-2)
