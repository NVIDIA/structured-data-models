# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete native column domains over row-owned cell storage."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from sdm.cache import KVCacheEntry
from sdm.execution._collectives import Exchange
from sdm.nn.attention import _chunk


@dataclass(frozen=True)
class Domain:
    """Closed inducing/output dependency interval in global column order."""

    start: int
    stop: int
    owner: int


def domain_rounds(
    columns: int, inducing_width: int, output_width: int, world: int
) -> tuple[tuple[Domain, ...], ...]:
    """Assign one complete common-boundary domain per rank and round."""
    if min(columns, inducing_width, output_width, world) < 1:
        raise ValueError("Column dependency dimensions must be positive")
    iw = min(inducing_width, 65535, columns)
    ow = min(output_width, 65535, columns)
    width = math.lcm(iw, ow)
    domains = tuple(
        (
            Domain(start, min(start + width, columns), index % world)
            for index, start in enumerate(range(0, columns, width))
        )
    )
    return tuple(
        domains[start : start + world]
        for start in range(0, len(domains), world)
    )


def to_column_owners(
    local: Tensor,
    lengths: tuple[int, ...],
    domains: tuple[Domain, ...],
    group: Exchange,
) -> dict[Domain, Tensor]:
    """Transpose row-owned cells to only assigned global column domains."""
    rank, world = (group.rank, group.world)
    channels = local.size(-1)
    owned = tuple(domain for domain in domains if domain.owner == rank)
    send_splits = [
        sum(d.stop - d.start for d in domains if d.owner == peer)
        * lengths[rank]
        * channels
        for peer in range(world)
    ]
    receive_splits = [
        sum(d.stop - d.start for d in owned) * length * channels
        for length in lengths
    ]

    def pack() -> Tensor:
        send = local.new_empty(sum(send_splits))
        cursor = 0
        for peer in range(world):
            for domain in domains:
                if domain.owner == peer:
                    part = local[domain.start : domain.stop].reshape(-1)
                    send[cursor : cursor + part.numel()].copy_(part)
                    cursor += part.numel()
                    del part
        return send

    send = group.attempt(pack)
    received = group.exchange(send, send_splits, receive_splits)
    del send

    def assemble(received: Tensor = received) -> dict[Domain, Tensor]:
        fulls = {
            d: local.new_empty(d.stop - d.start, sum(lengths), channels)
            for d in owned
        }
        cursor, offset = (0, 0)
        for length in lengths:
            for domain in owned:
                count = (domain.stop - domain.start) * length * channels
                fulls[domain][:, offset : offset + length].copy_(
                    received[cursor : cursor + count].view(
                        domain.stop - domain.start, length, channels
                    )
                )
                cursor += count
            offset += length
        return fulls

    fulls = group.attempt(assemble)
    del received
    return fulls


def to_row_owners(
    outputs: dict[Domain, Tensor],
    local_out: Tensor,
    lengths: tuple[int, ...],
    domains: tuple[Domain, ...],
    group: Exchange,
) -> None:
    """Return destination rows, releasing global domains before receive."""
    rank, world = (group.rank, group.world)
    channels = local_out.size(-1)
    owned = tuple(d for d in domains if d.owner == rank)
    send_splits = [
        sum(d.stop - d.start for d in owned) * length * channels
        for length in lengths
    ]
    receive_splits = [
        sum(d.stop - d.start for d in domains if d.owner == peer)
        * lengths[rank]
        * channels
        for peer in range(world)
    ]

    def pack() -> Tensor:
        send = local_out.new_empty(sum(send_splits))
        cursor, offset = (0, 0)
        for length in lengths:
            for domain in owned:
                part = outputs[domain][:, offset : offset + length].reshape(-1)
                send[cursor : cursor + part.numel()].copy_(part)
                cursor += part.numel()
                del part
            offset += length
        return send

    send = group.attempt(pack)
    outputs.clear()
    received = group.exchange(send, send_splits, receive_splits)
    del send

    def writeback() -> None:
        cursor = 0
        for peer in range(world):
            for domain in domains:
                if domain.owner == peer:
                    count = (
                        (domain.stop - domain.start) * lengths[rank] * channels
                    )
                    local_out[domain.start : domain.stop].copy_(
                        received[cursor : cursor + count].view(
                            domain.stop - domain.start, lengths[rank], channels
                        )
                    )
                    cursor += count

    group.attempt(writeback)


def _native_pitch(full: Tensor, batch: torch.Size, readouts: int) -> Tensor:
    columns, rows, channels = full.shape
    backing = full.new_empty((*batch[:-1], rows, columns + readouts, channels))
    view = backing[..., readouts:, :].transpose(-2, -3)
    view.copy_(full.view(*batch[:-1], columns, rows, channels))
    return view


def _column_forward(
    column: Any,
    inducing: Any,
    output: Any,
    *,
    lengths: tuple[int, ...],
    group: Exchange,
    readouts: int,
) -> Any:

    def forward(
        *,
        query: Tensor,
        key_value: Tensor,
        out: Tensor,
        return_key_value: bool = False,
        **kwargs: Any,
    ) -> Any:
        del key_value, kwargs
        rank, world = (group.rank, group.world)
        batch = query.shape[:-2]
        columns, total, channels = (
            math.prod(batch),
            sum(lengths),
            query.size(-1),
        )
        size, points = (query.element_size(), column.inducing_points)
        iw = min(
            column.inducing_block.auto_batch_size_limit(
                query.device, size, points.size(0), total
            ),
            65535,
        )
        ow = min(
            column.output_block.auto_batch_size_limit(
                query.device, size, total, points.size(0)
            ),
            65535,
        )
        rounds = domain_rounds(columns, iw, ow, world)
        local = query.reshape(columns, lengths[rank], channels)
        local_out = out.reshape(columns, lengths[rank], channels)
        kv_shape = (
            columns,
            points.size(0),
            column.output_block.attn.num_key_value_heads,
            column.output_block.attn.head_dim,
        )
        cache_pair = group.attempt(
            lambda: (
                (query.new_empty(kv_shape), query.new_empty(kv_shape))
                if return_key_value
                else None
            )
        )

        def compute(fulls: dict[Domain, Tensor]) -> None:
            for domain, full in fulls.items():
                # Native inducing queries and their residual remain FP32
                # under autocast, unlike the expanded cell buffers.
                induced = points.new_empty(
                    domain.stop - domain.start, points.size(0), channels
                )
                for start in range(domain.start, domain.stop, iw):
                    stop = min(start + iw, columns)
                    part = full[start - domain.start : stop - domain.start]
                    native = (
                        _native_pitch(part, batch, readouts)
                        if columns <= iw
                        else part
                    )
                    q = (
                        points
                        if columns <= iw
                        else _chunk(points, batch, 2, start, stop)
                    )
                    state_out = (
                        None
                        if start == 0 or columns <= iw
                        else induced[
                            start - domain.start : stop - domain.start
                        ]
                    )
                    state = inducing(
                        query=q,
                        key_value=native,
                        batch_size_limit=iw,
                        out=state_out,
                    )
                    induced[start - domain.start : stop - domain.start].copy_(
                        state.reshape(stop - start, points.size(0), channels)
                    )
                    del part, native, q, state_out, state
                for start in range(domain.start, domain.stop, ow):
                    stop = min(start + ow, columns)
                    part = full[start - domain.start : stop - domain.start]
                    native = (
                        _native_pitch(part, batch, readouts)
                        if columns <= ow
                        else part
                    )
                    state = induced[start - domain.start : stop - domain.start]
                    if columns <= ow:
                        state = state.view(*batch, points.size(0), channels)
                    native_has_out = (
                        columns <= ow or len(batch) == 1 or total == 1
                    )
                    result = output(
                        query=native,
                        key_value=state,
                        return_key_value=return_key_value,
                        batch_size_limit=ow,
                        out=native if native_has_out else None,
                    )
                    if return_key_value:
                        assert cache_pair is not None
                        result, kv = result
                        cache_pair[0][start:stop].copy_(
                            kv.key.reshape(stop - start, *kv_shape[1:])
                        )
                        cache_pair[1][start:stop].copy_(
                            kv.value.reshape(stop - start, *kv_shape[1:])
                        )
                        del kv
                    part.copy_(result.reshape(stop - start, total, channels))
                    del part, native, state, result
                del induced

        for domains in rounds:
            fulls = to_column_owners(local, lengths, domains, group)
            group.attempt(lambda fulls=fulls: compute(fulls))
            to_row_owners(fulls, local_out, lengths, domains, group)
            if return_key_value:
                assert cache_pair is not None
                for domain in domains:
                    for start in range(domain.start, domain.stop, ow):
                        stop = min(start + ow, columns)
                        for tensor in cache_pair:
                            if world > 1:
                                group.broadcast(
                                    tensor[start:stop], domain.owner
                                )
            del fulls
        if not return_key_value:
            return out
        assert cache_pair is not None
        key, value = cache_pair
        return (
            out,
            KVCacheEntry(
                key.view(*batch, *kv_shape[1:]),
                value.view(*batch, *kv_shape[1:]),
            ),
        )

    return forward
