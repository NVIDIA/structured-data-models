# ruff: noqa: D101, D102

from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Parameter, RMSNorm

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.tabfm.cell_embedding import (
    CellEmbedding,
    FourierNanIndicatorCellEmbedding,
)
from sdm.nn import (
    GatedLogScale,
    InducedTransformerBlock,
    LogScale,
    RotaryEmbedding,
)


class RowEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        group_size: int,
        num_frequencies: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        cell_embedding: Literal[
            "fourier", "fourier_nan_indicator"
        ] = "fourier",
        row_log_scale: bool = False,
        rope_fraction: float = 0.25,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.channels = channels

        if cell_embedding == "fourier":
            cell_embedding_cls = CellEmbedding
        else:
            assert cell_embedding == "fourier_nan_indicator"
            cell_embedding_cls = FourierNanIndicatorCellEmbedding
        self.cell_embedding = cell_embedding_cls(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, bias=False, **factory_kwargs)

        self.readout_token = Parameter(
            torch.empty(num_readout_tokens, channels, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        rope = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            theta=100_000,
            requires_grad=False,
            partial_rotary_factor=rope_fraction,
            **factory_kwargs,
        )

        self.col_blocks = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_inducing_points=num_inducing_points,
                inducing_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    query_scaling=LogScale(
                        num_heads=num_heads,
                        **factory_kwargs,
                    ),
                    **factory_kwargs,
                ),
                output_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    query_scaling=None,
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
                query_scaling=GatedLogScale(
                    channels=channels // num_heads,
                    num_heads=num_heads,
                    **factory_kwargs,
                )
                if row_log_scale
                else None,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R, K * D]

        R_train = y.size(-1)
        K = self.readout_token.size(-2)

        buffer = torch.empty(
            (*x.size()[:-1], K + x.size(-1), self.channels),
            device=x.device,
            dtype=torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype,
        )
        buffer[..., :K, :] = self.readout_token.to(buffer.dtype)
        if isinstance(
            self.cell_embedding,
            FourierNanIndicatorCellEmbedding,
        ):
            if cache is not None and cache.is_replaying:
                context_mean = cast(Tensor, cache["cell_embedding.mean"])
            else:
                context_mean = self.cell_embedding._context_mean(
                    x,
                    train_size=R_train,
                )
                if cache is not None:
                    cache["cell_embedding.mean"] = context_mean
            x = self.cell_embedding(
                x,
                categorical_mask,
                train_size=R_train,
                context_mean=context_mean,
                batch_size_limit="auto",
                out=buffer[..., K:, :],
            )
        else:
            x = self.cell_embedding(
                x,
                categorical_mask,
                batch_size_limit="auto",
                out=buffer[..., K:, :],
            )

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).unsqueeze(-2)  # [..., R_train, 1, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).unsqueeze(-2)
            x[..., :R_train, :, :] += y_emb.to(x.dtype)

        x = x.transpose(-2, -3)
        for i, (col_block, row_block) in enumerate(
            zip(self.col_blocks, self.row_blocks)
        ):
            key = f"row_embedding.col_block{i}"
            result = col_block(
                query=x,
                key_value=cast(KVCacheEntry, cache[key])
                if cache is not None and cache.is_replaying
                else x[..., :R_train, :],
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit="auto",
                out=x,
            )

            if cache is not None and cache.is_recording:
                cache[key] = result[1]
            del result

            buffer = row_block(
                query=buffer[..., :K, :]
                if i == len(self.row_blocks) - 1
                else buffer,
                key_value=buffer,
                batch_size_limit="auto",
                out=buffer[..., :K, :]
                if i == len(self.row_blocks) - 1
                else buffer,
            )

        return self.norm(buffer).flatten(-2)
