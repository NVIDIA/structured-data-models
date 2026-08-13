# ruff: noqa: D101, D102

import math
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList, Parameter

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabiclv2.block import TabICLv2TransformerBlock
from sdm.nn import InducedTransformerBlock, RotaryEmbedding
from sdm.nn.memory import (
    attention_batch_size_limit,
    cuda_attention_memory_limit,
)

_MEMORY_EFFICIENT_COLUMN_CHUNK_SIZE = 4
_MEMORY_EFFICIENT_ROW_CHUNK_SIZE = 2048


class RowEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        group_size: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        norm_bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.lin = Linear(group_size, channels, **factory_kwargs)
        self.num_heads = num_heads

        self.num_classes = num_classes
        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, **factory_kwargs)

        self.col_layers = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_inducing_points=num_inducing_points,
                inducing_block=TabICLv2TransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    norm_bias=norm_bias,
                    qassmax=True,
                    **factory_kwargs,
                ),
                output_block=TabICLv2TransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    norm_bias=norm_bias,
                    qassmax=False,
                    **factory_kwargs,
                ),
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.readout_token = Parameter(
            torch.empty((num_readout_tokens, channels), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        rope = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            theta=100_000,
            **factory_kwargs,
        )
        self.row_layers = ModuleList(
            TabICLv2TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                norm_bias=norm_bias,
                qassmax=False,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)

    def _project_chunk(
        self,
        x: Tensor,
        row_start: int,
        row_end: int,
        feature_index: Tensor,
        y_emb: Tensor | None,
        num_digits: int,
    ) -> Tensor:
        x = self.lin(x[..., row_start:row_end, :][..., feature_index])
        if num_digits > 1:
            if y_emb is None:
                x = x.unsqueeze(0).expand(num_digits, *x.size())
            else:
                x = x.unsqueeze(0).repeat(num_digits, *(1,) * x.dim())
        if y_emb is not None:
            target_end = min(row_end, y_emb.size(-3))
            if row_start < target_end:
                x[..., : target_end - row_start, :, :] += y_emb[
                    ..., row_start:target_end, :, :
                ].to(x.dtype)
        return x

    def _memory_efficient_forward(
        self,
        x: Tensor,
        y_emb: Tensor | None,
        *,
        R_train: int,
        num_digits: int,
        cache: Cache | None,
        batch_size_limit: int | None,
    ) -> Tensor:
        R, C = x.size()[-2:]
        G = self.lin.in_features

        def memory_limit() -> int | None:
            if x.device.type != "cuda":
                return None
            return cuda_attention_memory_limit(x.device)

        shift = 2 ** torch.arange(G, device=x.device)
        feature_index = torch.arange(C, device=x.device).view(C, 1)
        feature_index = (feature_index + shift.view(1, G)) % C

        layer_caches: list[KVCacheEntry]
        if cache is not None and cache.is_replaying:
            layer_caches = [
                cast(KVCacheEntry, cache[f"row_embedding.col_layer{i}"])
                for i in range(len(self.col_layers))
            ]
        else:
            cache_chunks: list[list[KVCacheEntry]] = [
                [] for _ in self.col_layers
            ]
            # Phase 1: build each layer's inducing K/V by column chunks.
            for start in range(0, C, _MEMORY_EFFICIENT_COLUMN_CHUNK_SIZE):
                end = min(
                    start + _MEMORY_EFFICIENT_COLUMN_CHUNK_SIZE,
                    C,
                )
                context = (
                    self._project_chunk(
                        x=x,
                        row_start=0,
                        row_end=R_train,
                        feature_index=feature_index[start:end],
                        y_emb=y_emb,
                        num_digits=num_digits,
                    )
                    .transpose(-2, -3)
                    .contiguous()
                )
                for i, layer in enumerate(self.col_layers):
                    layer = cast(InducedTransformerBlock, layer)
                    limit = attention_batch_size_limit(
                        requested_limit=batch_size_limit,
                        query=context,
                        key_value=context,
                        attention_memory_limit=memory_limit(),
                    )
                    if i < len(self.col_layers) - 1:
                        context, layer_cache = layer(
                            query=context,
                            key_value=context,
                            return_key_value=True,
                            batch_size_limit=limit,
                        )
                    else:
                        layer_cache = layer.induced_key_value(
                            context,
                            batch_size_limit=limit,
                        )
                    cache_chunks[i].append(layer_cache)

            layer_caches = [
                KVCacheEntry(
                    key=torch.cat([entry.key for entry in chunks], dim=-4),
                    value=torch.cat([entry.value for entry in chunks], dim=-4),
                )
                for chunks in cache_chunks
            ]
            if cache is not None:
                for i, layer_cache in enumerate(layer_caches):
                    cache[f"row_embedding.col_layer{i}"] = layer_cache

        out: Tensor | None = None
        # Phase 2: replay those K/V summaries over independent row chunks.
        for start in range(0, R, _MEMORY_EFFICIENT_ROW_CHUNK_SIZE):
            end = min(start + _MEMORY_EFFICIENT_ROW_CHUNK_SIZE, R)
            chunk = self._project_chunk(
                x=x,
                row_start=start,
                row_end=end,
                feature_index=feature_index,
                y_emb=y_emb,
                num_digits=num_digits,
            ).transpose(-2, -3)
            for layer, layer_cache in zip(
                self.col_layers, layer_caches, strict=True
            ):
                layer = cast(InducedTransformerBlock, layer)
                limit = attention_batch_size_limit(
                    requested_limit=batch_size_limit,
                    query=chunk,
                    key_value=layer_cache,
                    attention_memory_limit=memory_limit(),
                )
                chunk = layer(
                    query=chunk,
                    key_value=layer_cache,
                    batch_size_limit=limit,
                )
            if num_digits > 1:
                chunk = chunk.mean(dim=0)
            chunk = self._aggregate_rows(
                chunk,
                batch_size_limit=batch_size_limit,
                attention_memory_limit=memory_limit(),
            )
            if out is None:
                out = chunk.new_empty((*chunk.size()[:-2], R, chunk.size(-1)))
            out[..., start:end, :].copy_(chunk)

        assert out is not None
        return out

    def _aggregate_rows(
        self,
        x: Tensor,  # [..., C, R, D]
        *,
        batch_size_limit: int | None,
        attention_memory_limit: int | None,
    ) -> Tensor:
        *B, _, R, D = x.size()
        K = self.readout_token.size(-2)
        x = torch.cat(
            [
                self.readout_token.to(x.dtype)
                .view(*(1,) * len(B), 1, K, D)
                .expand(*B, R, K, D),
                x.transpose(-2, -3),
            ],
            dim=-2,
        )  # [..., R, K + C, D]
        row_limit = attention_batch_size_limit(
            requested_limit=batch_size_limit,
            query=x,
            key_value=x,
            attention_memory_limit=attention_memory_limit,
            num_heads=self.num_heads,
        )
        for i, row_layer in enumerate(self.row_layers):
            row_layer = cast(TabICLv2TransformerBlock, row_layer)
            query = x[..., :K, :] if i == len(self.row_layers) - 1 else x
            x = row_layer(
                query=query,
                key_value=x,
                batch_size_limit=row_limit,
            )
        return self.norm(x).view(*B, R, K * D)

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        *,
        train_mask: Tensor | None = None,  # [R],
        max_keys: int | None = None,
        num_classes: int | None = None,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
        generator: torch.Generator | None = None,
        memory_efficient: bool = False,
    ) -> Tensor:  # [..., R, K * D]
        R, C = x.size()[-2:]
        R_train = y.size(-1)
        G = self.lin.in_features
        train_index: Any = slice(R_train) if train_mask is None else train_mask
        plan_attention = (
            x.device.type == "cuda"
            and not self.training
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        )
        num_digits = 1
        if (
            self.y_emb is not None
            and num_classes is not None
            and num_classes > self.num_classes
        ):
            bases = _mixed_radix_bases(num_classes, self.num_classes)
            num_digits = len(bases)
            if y.numel() > 0:
                y = _mixed_radix_digits(y, bases)  # [F, ..., R_train]

        y_emb: Tensor | None = None
        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).unsqueeze(-2)
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).unsqueeze(-2)

        if memory_efficient and R > _MEMORY_EFFICIENT_ROW_CHUNK_SIZE:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "Memory-efficient row embedding does not support "
                    "torch.compile"
                )
            if self.training:
                raise RuntimeError(
                    "Memory-efficient row embedding requires evaluation mode"
                )
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "Memory-efficient row embedding requires gradients to "
                    "be disabled"
                )
            if not (
                isinstance(train_index, slice)
                and train_index.indices(R) == (0, R_train, 1)
            ):
                raise ValueError(
                    "Memory-efficient row embedding requires the training "
                    "rows to be a contiguous prefix"
                )
            if max_keys is not None:
                raise ValueError(
                    "Memory-efficient row embedding does not support "
                    "'max_keys'"
                )
            return self._memory_efficient_forward(
                x=x,
                y_emb=y_emb,
                R_train=R_train,
                num_digits=num_digits,
                cache=cache,
                batch_size_limit=batch_size_limit,
            )

        # Feature grouping: gather G columns into each token.
        shift = 2 ** torch.arange(G, device=x.device)
        index = torch.arange(C, device=x.device)
        index = (index.view(C, 1) + shift.view(1, G)) % C  # [C, G]
        x = x[..., index]  # [..., R, C, G]
        x = self.lin(x)  # [..., R, C, D]

        if num_digits > 1:
            if y.numel() > 0:
                x = x.unsqueeze(0).repeat(num_digits, *(1,) * x.dim())
            else:
                x = x.unsqueeze(0).expand(num_digits, *x.size())

        if y_emb is not None:
            # y_emb has shape [F, ..., R_train, 1, D]:
            x[..., train_index, :, :] += y_emb.to(x.dtype)

        # Column-wise induced set attention (B * C as the batch axis).
        # Materialize once to avoid repeated copies in the column layers.
        x = x.transpose(-2, -3).contiguous()  # [..., C, R, D]
        col_batch_size_limit = batch_size_limit
        for i, col_layer in enumerate(self.col_layers):
            col_layer = cast(InducedTransformerBlock, col_layer)
            key = f"row_embedding.col_layer{i}"
            if cache is not None and cache.is_replaying:
                key_value = cast(KVCacheEntry, cache[key])
            else:
                key_value = x[..., train_index, :]
                if max_keys is not None and key_value.size(-2) > max_keys:
                    index = torch.randperm(
                        key_value.size(-2),
                        device=key_value.device,
                        generator=generator,
                    )[:max_keys]
                    key_value = key_value[..., index, :]

            if i == 0 or (
                plan_attention and cache is not None and cache.is_recording
            ):
                col_batch_size_limit = attention_batch_size_limit(
                    requested_limit=batch_size_limit,
                    query=x,
                    key_value=key_value,
                    attention_memory_limit=(
                        cuda_attention_memory_limit(x.device)
                        if plan_attention
                        else None
                    ),
                )
            result = col_layer(
                query=x,  # [..., C, R, D]
                key_value=key_value,  # [..., C, R_train, D]
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit=col_batch_size_limit,
            )  # [..., C, R, D]

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

        if num_digits > 1:  # Average over mixed-radix digits.
            x = x.mean(dim=0)  # [F, ..., C, R, D] -> [..., C, R, D]

        return self._aggregate_rows(
            x,
            batch_size_limit=batch_size_limit,
            attention_memory_limit=(
                cuda_attention_memory_limit(x.device)
                if plan_attention
                else None
            ),
        )


def _mixed_radix_bases(num_classes: int, max_classes: int) -> list[int]:
    num_digits = math.ceil(math.log(num_classes) / math.log(max_classes))
    base = min(math.ceil(num_classes ** (1.0 / num_digits)), max_classes)
    bases = [base] * num_digits
    product = base**num_digits
    for i in range(num_digits):
        if product >= num_classes:
            break
        if bases[i] < max_classes:
            product = product // bases[i] * (bases[i] + 1)
            bases[i] += 1

    return bases


def _mixed_radix_digits(y: Tensor, bases: list[int]) -> Tensor:
    F = len(bases)
    divisors = [1] * F
    for i in range(F - 2, -1, -1):
        divisors[i] = divisors[i + 1] * bases[i + 1]

    size = (-1,) + (1,) * y.dim()
    divisor = torch.tensor(divisors, device=y.device).view(size)
    base = torch.tensor(bases, device=y.device).view(size)
    return (y.unsqueeze(0) // divisor) % base  # [F, ...]
