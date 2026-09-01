# ruff: noqa: D101, D102

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Parameter, RMSNorm

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import InducedTransformerBlock, RotaryEmbedding


class RowEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, **factory_kwargs)

        self.readout_token = Parameter(
            torch.empty(num_readout_tokens, channels, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        rope = RotaryEmbedding(
            channels=channels // num_heads,
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
                    num_heads=num_heads,
                    query_log_scale=True,
                    **factory_kwargs,
                ),
                output_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
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
                num_heads=num_heads,
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
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R, K * D]

        *B, R, _, D = x.size()
        R_train = y.size(-1)
        K = self.readout_token.size(-2)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).unsqueeze(-2)
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).unsqueeze(-2)
            x[..., :R_train, :, :] += y_emb.to(x.dtype)

        for i, (col_block, row_block) in enumerate(
            zip(self.col_blocks, self.row_blocks)
        ):
            x_i = x if i == 0 else x[..., K:, :]
            x_i = x_i.transpose(-2, -3).contiguous()

            key = f"row_embedding.col_block{i}"
            if cache is not None and cache.is_replaying:
                key_value = cast(KVCacheEntry, cache[key])
            else:
                key_value = x_i[..., :R_train, :]

            result = col_block(
                query=x_i,
                key_value=key_value,
                return_key_value=cache is not None and cache.is_recording,
            )

            if cache is not None and cache.is_recording:
                x_i, cache[key] = result
            else:
                x_i = result

            x_i = x_i.transpose(-2, -3)

            if i == 0:
                readout_token = (
                    self.readout_token.to(x_i.dtype)
                    .view(*(1,) * len(B), 1, K, D)
                    .expand(*B, R, K, D)
                )
                x = torch.cat([readout_token, x_i], dim=-2)
            else:
                x = torch.cat([x[..., :K, :], x_i], dim=-2)
            del x_i

            x = row_block(
                query=x[..., :K, :] if i == len(self.row_blocks) - 1 else x,
                key_value=x,
            )

        return self.norm(x).flatten(-2)
