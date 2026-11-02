# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match string categories by hashing whole labels and comparing bytes."""

from collections.abc import Callable, Sequence
from typing import NamedTuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import StringTensor

# Labels are hashed and compared in chunks of this many bytes.
_CHUNK = 64
# Eager code gathers at least this many bytes at once.
_BLOCK = 1 << 11


def _odd_constants(count: int, seed: int = 0) -> list[int]:
    """Return odd 64-bit constants as signed integers (SplitMix64)."""
    constants, state = [], seed
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) % (1 << 64)
        value = state
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 % (1 << 64)
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB % (1 << 64)
        value = (value ^ (value >> 31)) | 1
        constants.append(value - (1 << 64) if value >> 63 else value)
    return constants


_MIXERS = _odd_constants(2)


class _StringRows(NamedTuple):
    """Byte storage, start offsets and lengths of string labels.

    The storage ends with ``_CHUNK`` zero bytes, so chunks can start at any
    label byte.
    """

    data: Tensor
    starts: Tensor
    lengths: Tensor


class _StringIndex(NamedTuple):
    """Sorted hashes of distinct fitted labels, for binary search.

    Each fitted string tensor has a slot, and its labels are rows, joined
    slot by slot. ``hashes`` holds the sorted hashes of each label with its
    slot, ``rows`` the row of each hash, and ``starts`` the first row of
    each slot, then the number of rows. A label's code is its row minus its
    slot's start. For example, slots ``["blue", "red"]`` and ``["L", "M"]``
    give rows 0 to 3, ``starts == [0, 2, 4]``, and code 1 for row 3, "M".
    Labels are hashed with ``seed``, the first under which distinct labels
    do not share a hash.
    """

    hashes: Tensor
    rows: Tensor
    starts: Tensor
    seed: int


def _string_index(values: Sequence[StringTensor]) -> _StringIndex:
    """Hash the labels of fitted string tensors, one slot per tensor.

    Of equal labels of a slot, only the last is kept. This runs eagerly,
    when categories are fitted or loaded.
    """
    counts = torch.tensor(
        [value.numel() for value in values], dtype=torch.int64
    )
    starts = F.pad(counts.cumsum(0), (1, 0))
    if not values:
        empty = torch.empty(0, dtype=torch.int64)
        return _StringIndex(empty, empty, starts, seed=0)
    rows = _join_rows(values)
    device = rows.data.device
    slots = torch.arange(len(values), device=device).repeat_interleave(
        counts.to(device)
    )
    seed = 0
    while True:
        hashes, order = _hash_labels(rows, slots, seed).sort(stable=True)
        # Equal labels form runs of equal hashes; all labels of a run must
        # be equal, and only the last is kept.
        first = (hashes[1:] == hashes[:-1]).nonzero().squeeze(1)
        earlier = _select_rows(rows, order[first])
        later = _select_rows(rows, order[first + 1])
        equal = (slots[order[first]] == slots[order[first + 1]]) & (
            earlier.lengths == later.lengths
        )
        equal &= _labels_equal(
            earlier._replace(lengths=earlier.lengths.where(equal, 0)), later
        )
        if equal.all():
            break
        # Distinct labels share a hash; hash all labels again.
        seed += 1
    keep = torch.ones_like(hashes, dtype=torch.bool)
    keep[first] = False
    return _StringIndex(hashes[keep], order[keep], starts.to(device), seed)


def _native_string_lookups(
    left: tuple[StringTensor, ...],
    right: tuple[StringTensor, ...],
    slots: Sequence[int],
    index: _StringIndex,
    dtype: torch.dtype,
) -> tuple[Tensor, ...]:
    """Map each query vocabulary to IDs in its fitted vocabulary.

    ``right`` holds fitted tensors and ``slots`` their slots in ``index``.
    Missing labels map to ``-1``; the last of duplicate fitted labels wins.
    """
    # Broadcast queries are matched once.
    queries = tuple(_distinct_rows(value) for value in left)
    query = _join_rows(queries)
    device = query.data.device
    pairs = _pairs(queries, device)
    slot = torch.tensor(slots, device=device)
    hashes = _hash_labels(query, slot[pairs], index.seed)
    # Fitted hashes are distinct, so only the last one not above a query's
    # hash may match it.
    position = torch.searchsorted(index.hashes, hashes, right=True) - 1
    position = position.clamp_min(0)
    fitted = _join_rows(right)
    # Each slot's rows start here, and each fitted tensor's at `first` in
    # `fitted`.
    start = index.starts[slot]
    size = index.starts[slot + 1] - start
    first = torch.tensor(
        [0, *(value.numel() for value in right[:-1])],
        dtype=torch.int64,
        device=device,
    ).cumsum(0)
    # The candidate's code in its slot: its row minus the slot's start.
    code = index.rows[position] - start[pairs]
    # It may match only with the query's hash, within the query's slot.
    found = (
        (index.hashes[position] == hashes) & (code >= 0) & (code < size[pairs])
    )
    candidate = _select_rows(
        fitted, (first[pairs] + code).clamp(0, fitted.starts.numel() - 1)
    )
    found &= candidate.lengths == query.lengths
    # Only candidates' bytes are compared.
    compared = query._replace(lengths=query.lengths.where(found, 0))
    found &= _labels_equal(compared, candidate)
    output = code.where(found, -1)
    return tuple(
        part.to(dtype).expand(value.numel())
        for part, value in zip(
            output.split([value.numel() for value in queries]), left
        )
    )


def _distinct_rows(value: StringTensor) -> StringTensor:
    """Return one row of a broadcast view, or the view itself."""
    if value.stride(0) != 0:
        return value
    return cast(StringTensor, value[: min(value.numel(), 1)])


def _label_offsets(value: StringTensor) -> tuple[Tensor, Tensor]:
    """Return start and end offsets of the visible labels, as views."""
    offset, first = value._offset, value._logical_storage_offset
    count, step = value.numel(), value.stride(0)
    if step == 0:
        # A broadcast view repeats one label.
        return (
            offset[first : first + 1].expand(count),
            offset[first + 1 : first + 2].expand(count),
        )
    last = first + count * step
    return offset[first:last:step], offset[first + 1 : last + 1 : step]


def _pairs(values: tuple[StringTensor, ...], device: torch.device) -> Tensor:
    """Return the position in ``values`` of each joined label."""
    counts = torch.tensor([value.numel() for value in values], device=device)
    return torch.searchsorted(
        counts.cumsum(0),
        torch.arange(sum(value.numel() for value in values), device=device),
        right=True,
    )


def _join_rows(values: Sequence[StringTensor]) -> _StringRows:
    """Concatenate the visible labels of one-dimensional string tensors."""
    # Each tensor's offsets index its own bytes; shift them past earlier ones.
    starts, lengths, base = [], [], 0
    for value in values:
        start, end = _label_offsets(value)
        starts.append(start.long() + base)
        lengths.append((end - start).long())
        base += value._data.numel()
    return _StringRows(
        data=torch.cat(
            [
                *(value._data for value in values),
                values[0]._data.new_zeros(_CHUNK),
            ]
        ),
        starts=torch.cat(starts),
        lengths=torch.cat(lengths),
    )


def _select_rows(rows: _StringRows, index: Tensor) -> _StringRows:
    """Select labels of joined rows, which keep their byte storage."""
    return rows._replace(
        starts=rows.starts[index], lengths=rows.lengths[index]
    )


def _label_values(
    lengths: Tensor,
    labels: Sequence[_StringRows],
    value: Callable[..., Tensor],
) -> Tensor:
    """Sum ``value(offset, *chunks)`` over the chunks of each label.

    ``chunks`` hold ``_CHUNK`` bytes of each of ``labels`` at a chunk's
    ``offset`` in its label. The last chunk of a label ends at its end,
    overlapping the one before, so only labels shorter than a chunk have
    bytes past their end; these are zero. Chunk counts are bounded by tensor
    sizes in compiled code, so graphs do not depend on label lengths; chunks
    past the last label read an empty label. Compiled code evaluates all
    chunks at once, fusing the gathers into ``value``; eager code evaluates
    blocks of chunks, keeping temporaries small. When no label is longer
    than a chunk, eager code reads one chunk per label, cut to the longest
    label; ``value`` must be zero on a zero chunk at offset zero, which
    stands for an empty label.
    """
    count = lengths.numel()
    width, label, offset = _CHUNK, None, 0
    longest = _CHUNK + 1
    if not torch.compiler.is_compiling() and count:
        longest = int(lengths.max())
    if longest > _CHUNK:
        # One empty label at the end takes the unused chunks.
        lengths = F.pad(lengths, (0, 1))
        counts = (lengths + (_CHUNK - 1)) // _CHUNK
        ends = counts.cumsum(0)
        if torch.compiler.is_compiling():
            # Distinct labels have at most this many chunks.
            total = (labels[0].data.numel() - _CHUNK) // _CHUNK + count
            mask = True
        else:
            short = (lengths > 0) & (lengths < _CHUNK)
            total, shorts = torch.stack([ends[-1], short.sum()]).tolist()
            mask = shorts > 0
        chunk = torch.arange(total, device=lengths.device)
        label = torch.searchsorted(ends, chunk, right=True).clamp_max(count)
        offset = torch.minimum(
            (chunk - ends[label] + counts[label]) * _CHUNK,
            (lengths[label] - _CHUNK).clamp_min(0),
        )
        starts = [
            F.pad(rows.starts, (0, 1))[label] + offset for rows in labels
        ]
    else:
        width = max(8, 1 << (longest - 1).bit_length())
        total, mask = count, True
        starts = [rows.starts for rows in labels]
    position = torch.arange(width, device=lengths.device)

    def evaluate(part: slice) -> Tensor:
        chunks = [
            rows.data.unfold(0, width, 1).index_select(0, start[part])
            for rows, start in zip(labels, starts, strict=True)
        ]
        if mask:
            length = lengths[part] if label is None else lengths[label[part]]
            past = position >= length[:, None]
            chunks = [chunk.masked_fill_(past, 0) for chunk in chunks]
        if label is None:
            return value(0, *chunks)
        return value(cast(Tensor, offset)[part], *chunks)

    if torch.compiler.is_compiling():
        values = evaluate(slice(None))
    else:
        # Blocks gather 16 bytes per label of one chunk, or 8 bytes per
        # chunk of longer labels, which keep more arrays per chunk.
        step = -(-max(_BLOCK, total * (16 if label is None else 8)) // width)
        blocks = [
            evaluate(slice(start, start + step))
            for start in range(0, max(total, 1), step)
        ]
        values = blocks[0] if len(blocks) == 1 else torch.cat(blocks)
    if label is None:
        return values
    sums = F.pad(values.cumsum(0), (1, 0))
    return sums[ends[:count]] - sums[ends[:count] - counts[:count]]


def _mix(value: Tensor) -> Tensor:
    """Mix the bits of 64-bit integers (MurmurHash3's finalizer)."""

    def shift(value: Tensor) -> Tensor:
        # A logical right shift by 33 bits.
        return (value >> 33) & ((1 << 31) - 1)

    for mixer in _MIXERS:
        value = (value ^ shift(value)) * mixer
    return value ^ shift(value)


def _hash_labels(rows: _StringRows, slots: Tensor, seed: int) -> Tensor:
    """Hash all bytes, the length and the slot of each label with ``seed``.

    A chunk's hash weights its signed 32-bit words, then mixes in its
    offset; compiled code weights bytes instead, with the same result.
    """
    # Weights of the 32-bit words of a chunk, then of chunk offsets, label
    # lengths and slots. Narrow words keep changes of high bits from
    # cancelling.
    *word_weights, offset_weight, length_weight, slot_weight = _odd_constants(
        _CHUNK // 4 + 3, seed
    )
    words = torch.tensor(word_weights, device=rows.data.device)

    def weigh(offset: Tensor | int, chunk: Tensor) -> Tensor:
        if not torch.compiler.is_compiling():
            # Eager chunks may be narrower; missing words are zero.
            value = chunk.view(torch.int32) * words[: chunk.size(-1) // 4]
        else:
            place = torch.arange(_CHUNK, device=words.device) % 4
            value = chunk.long()
            # The last byte of a word carries its sign.
            value = value - (value >> 7 << 8) * (place == 3)
            value = value * (words.repeat_interleave(4) << 8 * place)
        return _mix(value.sum(-1) + offset * offset_weight)

    return _mix(
        _label_values(rows.lengths, (rows,), weigh)
        + rows.lengths * length_weight
        + slots * slot_weight
    )


def _labels_equal(rows: _StringRows, other: _StringRows) -> Tensor:
    """Compare paired labels byte by byte, up to the lengths of ``rows``."""

    def differ(_: object, chunk: Tensor, other: Tensor) -> Tensor:
        if not torch.compiler.is_compiling():
            # Eager code compares words.
            chunk, other = chunk.view(torch.int64), other.view(torch.int64)
        return (chunk != other).any(-1).long()

    return _label_values(rows.lengths, (rows, other), differ) == 0
