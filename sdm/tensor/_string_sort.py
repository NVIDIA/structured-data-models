# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Exact stable string sorting with native PyTorch operations.

Strings are compared as integer words. Each word holds seven bytes of a
string, first byte in the highest bits, so comparing words compares bytes
in order. Each byte is stored as ``byte + 1`` (1 to 256, nine bits) and 0
means "the string has ended", so ``"a"`` (``[98, 0, ...]``) sorts before
``"a\0"`` (``[98, 1, ...]``), and a prefix before its extensions. Seven
nine-bit bytes use 63 bits, keeping the ``int64`` non-negative so integer
order matches byte order. Null strings are treated as empty and placed last
separately.

An LSD radix sort orders strings by their first ``PREFIX`` bytes while
tracking prefix ties. Longer strings that tie on the prefix are ranked by
their remaining bytes with block-pair prefix doubling. The same static-shape
code runs eagerly and under ``torch.compile``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from sdm.tensor.string import StringTensor

PREFIX = 21


def argsort(values: StringTensor, descending: bool) -> Tensor:
    """Return a stable sort permutation of one-dimensional strings.

    Nulls sort last in both directions.
    """
    n = values.numel()
    if values.stride(0) == 0:
        # Every row refers to the same string.
        return torch.arange(n, device=values.device)

    dtype = _index_dtype(max(values._offset.numel(), values._data.numel() + 1))
    rows = torch.arange(n, device=values.device, dtype=dtype)
    rows = rows * values.stride(0) + values._logical_storage_offset
    start = values._offset[rows].to(dtype)
    length = values._offset[rows + 1].to(dtype) - start
    valid = None if values._valid is None else values._valid[rows]
    if valid is not None:
        # Null payloads compare as empty and keep their input order.
        length = length.where(valid, 0)
    # Masked bytes read any valid position, so empty buffers get one byte.
    data = values._data if values._data.numel() else values._data.new_zeros(1)
    bound = values._max_length
    if bound is None:
        # Round the byte count up to 2**k - 1 with comparisons only, so
        # compiled code reuses one graph per bucket instead of per size.
        bound = 0
        while bound < values._data.numel():
            bound = 2 * bound + 1

    # LSD radix sort over the prefix, least significant word first.
    prefix = min(max(bound, 1), PREFIX)
    perm = rank = None
    for depth in reversed(range(0, prefix, 7)):
        rows = slice(None) if perm is None else perm
        key = _word(
            data, start[rows], length[rows], depth, min(7, prefix - depth)
        )
        if depth == 0 and valid is not None:
            key = key.where(
                valid[rows], -1 if descending else torch.iinfo(key.dtype).max
            )
        key, order = key.sort(descending=descending, stable=True)
        perm = order if perm is None else perm[order]
        if bound > prefix:
            # Rank the prefix bytes processed so far in sorted order.
            changed = torch.ones_like(key, dtype=torch.bool)
            changed[1:] = key[1:] != key[:-1]
            if rank is not None:
                rank = rank[order]
                changed[1:] |= rank[1:] != rank[:-1]
            rank = changed.cumsum(0)
    assert perm is not None  # The prefix spans at least one word.
    if rank is None:
        return perm

    # Only longer strings that tie on the prefix need their remaining bytes.
    tied = torch.zeros_like(rank, dtype=torch.bool)
    tied[1:] = rank[1:] == rank[:-1]
    tied[:-1] |= tied[1:].clone()
    tied &= length[perm] > prefix
    refine = functools.partial(_refine, bound=bound, descending=descending)
    if torch.compiler.is_compiling():
        # Static bounds: at most total // (PREFIX + 1) strings are longer,
        # and their tails span at most total // 7 seven-byte blocks.
        total = values._data.numel()
        slots = min(n, total // (PREFIX + 1))
        blocks = min(total // 7, slots * -(-(bound - PREFIX) // 7))
        refine = functools.partial(refine, slots=slots, blocks=blocks)
        operands = (perm, rank, tied, data, start, length)
        return torch.cond(
            tied.any(), refine, lambda perm, *_: perm.clone(), operands
        )

    # Eager: groups of identical strings are already in order. Each pair is
    # two neighbors in a prefix group, at least one longer than the prefix.
    pair = (rank[1:] == rank[:-1]) & (tied[1:] | tied[:-1])
    pair = pair.nonzero().squeeze(1)
    if pair.numel() == 0:
        return perm
    tied &= _mixed(data, start, length, perm, rank, pair)
    del pair
    if not tied.any():
        return perm
    blocks = ((length[perm] - PREFIX + 6) // 7)[tied]
    slots, blocks = int(tied.sum()), int(blocks.sum())
    return refine(
        perm, rank, tied, data, start, length, slots=slots, blocks=blocks
    )


def _refine(
    perm: Tensor,
    rank: Tensor,
    tied: Tensor,
    data: Tensor,
    start: Tensor,
    length: Tensor,
    *,
    bound: int,
    descending: bool,
    slots: int,
    blocks: int,
) -> Tensor:
    """Order prefix ties by their remaining bytes.

    ``slots`` and ``blocks`` bound the tied strings and their tail blocks.
    """
    n = perm.numel()
    # A constant fill value keeps the compiled graph independent of ``n``.
    index = torch.nonzero_static(tied, size=slots, fill_value=-1).squeeze(1)
    used = index >= 0
    rows = perm[index.clamp_min(0)]
    tail = _block_rank(
        data=data,
        start=start[rows] + PREFIX,
        length=(length[rows] - PREFIX).where(used, 0),
        bound=bound - PREFIX,
        capacity=blocks,
    )
    # Padding slots scatter to their own destinations past the rows.
    padding = n + torch.arange(slots, device=data.device)
    tail = tail.new_zeros(n + slots).scatter_(
        0, index.where(used, padding), tail
    )[:n]
    if descending:
        tail = slots - tail
    # Tail ranks are at most ``slots``, so the combined key stays ordered.
    key = rank * (slots + 1) + tail
    return perm[key.argsort(stable=True)]


def _mixed(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    perm: Tensor,
    rank: Tensor,
    pair: Tensor,
) -> Tensor:
    """Whether each position's prefix group holds different strings (eager).

    A group is identical exactly when each string equals its predecessor, so
    only the neighbors at ``pair`` and ``pair + 1`` are compared. Bytes are
    compared only in groups of equal lengths, in four chunks so temporaries
    stay near the data size.
    """
    row, prev = perm[pair + 1], perm[pair]
    group = rank[pair]
    count = length[row]
    mixed = torch.zeros(rank.numel() + 1, dtype=torch.bool, device=rank.device)
    mixed[group[count != length[prev]]] = True
    count = (count - PREFIX).where(~mixed[group], 0)
    row, prev = start[row] + PREFIX, start[prev] + PREFIX
    dtype = start.dtype
    first = torch.cat((count.new_zeros(1), count.cumsum(0, dtype=dtype)))
    total = int(first[-1])
    step = max(1, -(-total // 4))
    for begin in range(0, total, step):
        byte = torch.arange(
            begin, min(begin + step, total), device=data.device, dtype=dtype
        )
        owner = torch.searchsorted(
            first, byte, right=True, out_int32=dtype == torch.int32
        ).sub_(1)
        byte -= first[owner]
        mismatch = data[row[owner] + byte] != data[prev[owner] + byte]
        mixed[group[owner[mismatch]]] = True
    return mixed[rank]


def _word(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    depth: int,
    width: int = 7,
) -> Tensor:
    # One word per row for bytes [depth, depth + width): byte + 1 where the
    # string has a byte, 0 past its end. Three bytes fit in int32.
    dtype = torch.int32 if width <= 3 else torch.int64
    byte = torch.arange(
        depth, depth + width, device=data.device, dtype=start.dtype
    )
    absent = byte >= length[:, None]
    index = start[:, None] + byte  # [rows, width]
    index.masked_fill_(absent, data.numel() - 1)
    symbol = data[index]
    del index
    symbol = symbol.to(dtype).add_(1).masked_fill_(absent, 0)
    symbol <<= torch.arange(9 * (width - 1), -1, -9, device=data.device)
    return symbol.sum(-1, dtype=dtype)


def _index_dtype(size: int) -> torch.dtype:
    # Positions use int32 unless the indexed extent needs int64.
    return torch.int32 if size < torch.iinfo(torch.int32).max else torch.int64


def _layout(count: Tensor, capacity: int) -> tuple[Tensor, Tensor, Tensor]:
    """Map ``capacity`` slots to rows holding ``count`` slots each.

    Unused trailing slots belong to a padding row with index ``count.numel()``.
    """
    dtype = _index_dtype(capacity + 1)
    count = count.to(dtype)
    first = torch.cat((count.new_zeros(1), count.cumsum(0, dtype=dtype)))
    rows = torch.arange(count.numel() + 1, device=count.device, dtype=dtype)
    repeats = torch.cat((count, (capacity - first[-1]).reshape(1)))
    owner = rows.repeat_interleave(repeats, output_size=capacity)
    local = torch.arange(capacity, device=count.device, dtype=dtype)
    return first, owner, local - first[owner]


def _dense_rank(key: Tensor) -> Tensor:
    # Rank zero is reserved for key zero, which marks an empty block. One
    # extra trailing empty block gives out-of-row reads a zero rank.
    key = torch.cat((key, key.new_zeros(1)))
    sorted_key, order = key.sort()
    dtype = _index_dtype(key.numel())
    step = (sorted_key.diff() != 0).to(dtype).cumsum(0, dtype=dtype)
    step = torch.cat((step.new_zeros(1), step))
    return torch.empty_like(step).scatter_(0, order, step)


def _block_rank(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    bound: int,
    capacity: int,
) -> Tensor:
    """Rank rows by merging adjacent seven-byte blocks with static shapes.

    ``capacity`` must cover the total number of blocks.
    """
    n = start.numel()
    # Rows with an empty tail (padding) own no blocks.
    count = (length + 6) // 7
    first, owner, local = _layout(count, capacity)
    start = torch.cat((start, start.new_zeros(1)))
    length = torch.cat((length, length.new_zeros(1)))
    byte = start[owner] + 7 * local.to(start.dtype)
    # Bytes left in the row from this block; `byte + i < end` could overflow.
    rest = (start + length)[owner] - byte
    word = torch.zeros(capacity, dtype=torch.int64, device=data.device)
    for i in range(7):
        present = i < rest
        symbol = data[(byte + i).where(present, data.numel() - 1)].long()
        word |= (symbol + 1).where(present, 0) << (54 - 9 * i)
    rank = _dense_rank(word)

    # [blocks] -> [ceil(blocks / 2)] per row until one block remains.
    for _ in range(((bound - 1) // 7).bit_length()):
        merged = (count + 1) // 2
        merged_capacity = (capacity + n) // 2
        merged_first, owner, local = _layout(merged, merged_capacity)
        count = torch.cat((count, count.new_zeros(1)))
        left = (first[owner] + 2 * local).where(owner < n, capacity)
        right = (left + 1).where(2 * local + 1 < count[owner], capacity)
        rank = _dense_rank(rank[left].long() * (capacity + 1) + rank[right])
        count, capacity, first = merged, merged_capacity, merged_first
    return rank[first[:-1]]
