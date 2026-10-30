# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match string categories with compact keys and exact byte comparison."""

from functools import partial
from typing import NamedTuple, cast

import torch
from torch import Tensor

from sdm import StringTensor

# Keys hold at most this many words of a label. Longer labels contribute
# their beginning and end, and are compared in windows of as many bytes.
_MAX_KEY_WORDS = 16
_WINDOW = 8 * _MAX_KEY_WORDS
# A step of the lookup compares at most this many windows of a long label.
_MAX_WINDOWS = 8


class _StringRows(NamedTuple):
    """Byte storage, start offsets, lengths and pair index of string labels."""

    data: Tensor
    starts: Tensor
    lengths: Tensor
    groups: Tensor


class _LookupKeys(NamedTuple):
    """Flat packed query keys and sorted fitted hashes for candidate lookup.

    ``fitted_order`` maps sorted-hash positions back to fitted labels.
    """

    query: Tensor
    query_hashes: Tensor
    sorted_hashes: Tensor
    fitted_order: Tensor


def _native_string_lookups(
    left: tuple[StringTensor, ...],
    right: tuple[StringTensor, ...],
    dtype: torch.dtype = torch.int32,
) -> tuple[Tensor, ...]:
    """Map each query vocabulary to IDs in its fitted vocabulary.

    Missing labels map to ``-1``; the last of duplicate fitted labels wins.
    Keys hold the pair index, so all pairs are matched at once. Keys also
    hold the label length, so a query longer than every fitted label
    matches none: keys are sized from fitted labels alone, and compiled
    graphs do not depend on query lengths.
    """
    if not left:
        return ()
    if sum(fitted.numel() for fitted in right) == 0:
        return tuple(
            torch.full(query.shape, -1, dtype=dtype, device=query.device)
            for query in left
        )
    # Keys fit the longest fitted label, up to _MAX_KEY_WORDS; longer labels
    # also compare byte windows. Without bounds, keys take the full width.
    bounds = [fitted._max_length for fitted in right]
    known = None not in bounds
    key_bytes = max(cast(list[int], bounds)) if known else _WINDOW + 1
    key_words = _ceil_div(key_bytes, 8, _MAX_KEY_WORDS)
    windows = 0
    if key_bytes > key_words * 8:
        windows = _ceil_div(key_bytes, _WINDOW, _MAX_WINDOWS)
    extent = sum(
        value._data.numel() + value._offset.numel()
        for value in (*left, *right)
    )
    index_dtype = (
        torch.int32
        if extent + _WINDOW < torch.iinfo(torch.int32).max
        else torch.int64
    )
    # Broadcast queries are matched once. Broadcast fitted rows stay
    # repeated, so the last wins as for any duplicate.
    queries = tuple(_distinct_rows(value) for value in left)
    # One join holds the query labels, then the fitted labels; a tensor's
    # position modulo the number of pairs is its pair index.
    rows = _join_rows((*queries, *right), index_dtype)
    rows = rows._replace(groups=rows.groups % len(left))
    query_count = sum(value.numel() for value in queries)
    output = _match_string_keys(
        rows=rows,
        query_count=query_count,
        key_words=key_words,
        windows=windows,
        dtype=torch.int64,
    )
    # Fitted positions are global; subtract each pair's first position.
    first = torch.tensor(
        [0, *(value.numel() for value in right[:-1])],
        dtype=torch.int64,
        device=output.device,
    ).cumsum(0)
    output = torch.where(
        output >= 0, output - first[rows.groups[:query_count]], -1
    )
    return tuple(
        part.to(dtype).expand(value.numel())
        for part, value in zip(
            output.split([value.numel() for value in queries]), left
        )
    )


def _ceil_div(value: int, step: int, limit: int) -> int:
    """Return ``ceil(value / step)``, at least one and at most ``limit``.

    Comparisons specialize compiled code on a symbolic ``value``.
    """
    count = 1
    while count < limit and value > count * step:
        count += 1
    return count


def _distinct_rows(value: StringTensor) -> StringTensor:
    """Return one row of a broadcast view, or the view itself."""
    if value.stride(0) != 0:
        return value
    return cast(StringTensor, value[: min(value.numel(), 1)])


def _join_rows(
    values: tuple[StringTensor, ...], index_dtype: torch.dtype
) -> _StringRows:
    """Concatenate the visible labels of one-dimensional string tensors.

    Offset slices are views, so the work does not grow with the number of
    tensors. Padding lets byte windows start at any label.
    """
    starts, ends = [], []
    for value in values:
        offset, first = value._offset, value._logical_storage_offset
        count, step = value.numel(), value.stride(0)
        if step == 0:
            # A broadcast view repeats one label.
            starts.append(offset[first : first + 1].expand(count))
            ends.append(offset[first + 1 : first + 2].expand(count))
        else:
            last = first + count * step
            starts.append(offset[first:last:step])
            ends.append(offset[first + 1 : last + 1 : step])
    device = values[0].device
    counts = torch.tensor(
        [value.numel() for value in values], dtype=index_dtype, device=device
    ).cumsum(0)
    groups = torch.searchsorted(
        counts,
        torch.arange(sum(value.numel() for value in values), device=device),
        right=True,
        out_int32=index_dtype == torch.int32,
    )
    # Each tensor's offsets index its own bytes; shift them past earlier ones.
    bases = torch.tensor(
        [0, *(value._data.numel() for value in values[:-1])],
        dtype=index_dtype,
        device=device,
    ).cumsum(0, dtype=index_dtype)
    start = torch.cat(starts).to(index_dtype)
    return _StringRows(
        data=torch.cat(
            [
                *(value._data for value in values),
                values[0]._data.new_zeros(_WINDOW),
            ]
        ),
        starts=start + bases[groups],
        lengths=torch.cat(ends).to(index_dtype) - start,
        groups=groups,
    )


def _select_rows(rows: _StringRows, index: Tensor | slice) -> _StringRows:
    """Select labels of joined rows, which keep their byte storage."""
    return _StringRows(
        data=rows.data,
        starts=rows.starts[index],
        lengths=rows.lengths[index],
        groups=rows.groups[index],
    )


def _match_string_keys(
    rows: _StringRows,
    query_count: int,
    key_words: int,
    windows: int,
    dtype: torch.dtype,
) -> Tensor:
    """Return the fitted row of each query label, or ``-1``.

    The first ``query_count`` rows are query labels, the others fitted
    labels. Only fitted hashes are kept; candidates are packed again when
    compared. Labels longer than their keys compare ``windows`` byte
    windows per step; zero means every fitted label fits its key.
    """
    weights = torch.arange(
        start=1,
        end=2 * key_words + 4,
        step=2,
        dtype=torch.int64,
        device=rows.data.device,
    )
    # Hashes filter candidates; packed keys and bytes establish equality.
    hashes = _hash_string_keys(rows, key_words, weights)
    sorted_hashes, fitted_order = hashes[query_count:].sort(stable=True)
    keys = _LookupKeys(
        # Flat buffers keep a fixed layout across higher-order control flow.
        query=_pack_string_keys(
            _select_rows(rows, slice(query_count)), key_words
        ).flatten(),
        query_hashes=hashes[:query_count],
        sorted_hashes=sorted_hashes,
        fitted_order=fitted_order,
    )
    return _resolve_candidates(
        rows=rows,
        query_count=query_count,
        keys=keys,
        key_words=key_words,
        windows=windows,
        dtype=dtype,
    )


def _key_parts(key_words: int) -> list[tuple[int, int]]:
    """Return the word ranges reading a label's beginning and end."""
    if key_words == 1:
        return [(0, 1)]
    middle = (key_words + 1) // 2
    return [(0, middle), (middle, key_words)]


def _key_bytes(
    rows: _StringRows, key_words: int, first: int, last: int
) -> Tensor:
    """Return key words ``first`` to ``last`` of each label as bytes.

    The second part of a key reads the end of a label. Bytes past a label's
    end are zero. Each part is one gather of byte windows.
    """
    starts = rows.starts + 8 * first
    if first > 0:
        starts = starts + (rows.lengths - 8 * key_words).clamp_min(0)
    window = rows.data.unfold(0, 8 * (last - first), 1).index_select(0, starts)
    position = torch.arange(
        start=8 * first,
        end=8 * last,
        dtype=rows.lengths.dtype,
        device=rows.data.device,
    )
    return window.masked_fill_(position >= rows.lengths[:, None], 0)


def _pack_string_keys(rows: _StringRows, key_words: int) -> Tensor:
    """Pack lookup keys into ``[N, key_words + 2]`` integers.

    The length column distinguishes NUL bytes from padding, and the last
    column holds the pair index. Complete keys prove equality; keys of long
    labels require byte comparison.
    """
    return torch.cat(
        (
            *(
                _key_bytes(rows, key_words, first, last).view(torch.int64)
                for first, last in _key_parts(key_words)
            ),
            rows.lengths[:, None].long(),
            rows.groups[:, None].long(),
        ),
        dim=1,
    )


def _hash_string_keys(
    rows: _StringRows, key_words: int, weights: Tensor
) -> Tensor:
    """Hash packed keys, one part of their words at a time.

    Compiled code computes the same hash from bytes, so the gather fuses
    into the sum. Eager code hashes chunks of rows whose byte windows hold
    at most half the bytes of the joined rows.
    """
    hashes = (
        rows.lengths.long() * weights[key_words]
        + rows.groups.long() * weights[-1]
    )
    count = rows.starts.numel()
    for first, last in _key_parts(key_words):
        if torch.compiler.is_compiling():
            places = 1 << torch.arange(0, 64, 8, device=weights.device)
            part_weights = (weights[first:last, None] * places).flatten()
            hashes += (
                _key_bytes(rows, key_words, first, last).long() * part_weights
            ).sum(-1)
            continue
        size = rows.data.numel() + 3 * rows.starts.nbytes
        step = max(1, size // (16 * (last - first)))
        for start in range(0, count, step):
            chunk = _select_rows(rows, slice(start, start + step))
            hashes[start : start + step] += (
                _key_bytes(chunk, key_words, first, last).view(torch.int64)
                * weights[first:last]
            ).sum(-1)
    return hashes


def _resolve_candidates(
    rows: _StringRows,
    query_count: int,
    keys: _LookupKeys,
    key_words: int,
    windows: int,
    dtype: torch.dtype,
) -> Tensor:
    """Find exact fitted IDs, returning ``-1`` for missing query labels.

    Equal-hash candidates are visited from right to left, so the last fitted
    duplicate wins.
    """
    position = _candidate_or_none(
        torch.searchsorted(keys.sorted_hashes, keys.query_hashes, right=True)
        - 1,
        keys,
    )
    # Loops capture whole rows and select within them, since higher-order
    # operators reject captured views of one another.
    advance = partial(
        _advance_candidates,
        rows=rows,
        query_count=query_count,
        keys=keys,
        key_words=key_words,
        windows=windows,
    )
    # Most queries resolve at their first candidate, before the loop. Bytes
    # of long labels are compared in the loop.
    state = (position,)
    if windows:
        state += (torch.zeros_like(position),)
    state = advance(*state, compare=False)
    if torch.compiler.is_compiling():
        position = torch.while_loop(_has_pending_candidates, advance, state)[0]
    else:
        # Structured ops invoke Dynamo in eager; use ordinary control flow.
        while bool(_has_pending_candidates(*state)):
            state = advance(*state)
        position = state[0]
    return torch.where(
        position < -1, keys.fitted_order[(-2 - position).clamp_min(0)], -1
    ).to(dtype)


def _candidate_or_none(position: Tensor, keys: _LookupKeys) -> Tensor:
    """Return sorted-hash positions holding the query hash, otherwise -1."""
    return position.where(
        (position >= 0)
        & (keys.sorted_hashes[position.clamp_min(0)] == keys.query_hashes),
        -1,
    )


def _advance_candidates(
    position: Tensor,
    *offset: Tensor,
    rows: _StringRows,
    query_count: int,
    keys: _LookupKeys,
    key_words: int,
    windows: int,
    compare: bool = True,
) -> tuple[Tensor, ...]:
    """Test current candidates and step backward for unresolved queries.

    ``position`` is -1 for a missing query and ``-2 - p`` for one resolved
    at sorted-hash position ``p``. ``offset``, if given, holds the bytes
    compared so far of long labels whose keys match; these compare
    ``windows`` windows per step, unless ``compare`` is false.
    """
    query = _select_rows(rows, slice(query_count))
    pending = position >= 0
    candidates = _select_rows(
        rows, keys.fitted_order[position.clamp_min(0)] + query_count
    )
    query_keys = keys.query.view(query_count, key_words + 2)
    if torch.compiler.is_compiling():
        # Packing fuses into the comparison.
        matched = pending & (
            query_keys == _pack_string_keys(candidates, key_words)
        ).all(-1)
    else:
        # Compare one part at a time, holding no packed candidate keys.
        matched = (
            pending
            & (query_keys[:, key_words] == candidates.lengths)
            & (query_keys[:, key_words + 1] == candidates.groups)
        )
        for first, last in _key_parts(key_words):
            part = _key_bytes(candidates, key_words, first, last)
            matched &= (
                query_keys[:, first:last] == part.view(torch.int64)
            ).all(-1)
            del part
    comparing = torch.zeros_like(matched)
    if offset:
        comparing = matched & (query.lengths > 8 * key_words)
        if compare:
            end = offset[0] + windows * _WINDOW
            matched = matched & ~(
                comparing
                & _windows_differ(query, candidates, offset[0], windows)
            )
            comparing = comparing & matched & (end < query.lengths)
            offset = (torch.where(comparing, end, 0),)
    position = torch.where(
        matched,
        position.where(comparing, -2 - position),
        _candidate_or_none(position - 1, keys).where(pending, position),
    )
    return position, *offset


def _windows_differ(
    query: _StringRows, fitted: _StringRows, offset: Tensor, windows: int
) -> Tensor:
    """Compare ``windows`` windows of query and fitted label bytes.

    Comparison starts ``offset`` bytes into a label and stops at the query
    label's end. Temporaries hold one window per query label.
    """
    position = torch.arange(_WINDOW, dtype=offset.dtype, device=offset.device)
    query_windows = query.data.unfold(0, _WINDOW, 1)
    fitted_windows = fitted.data.unfold(0, _WINDOW, 1)
    # Windows past either label's end start at its end, staying in bounds.
    end = torch.minimum(query.lengths, fitted.lengths)
    differ = torch.zeros_like(end, dtype=torch.bool)
    for _ in range(windows):
        start = torch.minimum(offset, end)
        differ |= (
            (
                query_windows.index_select(0, query.starts + start)
                != fitted_windows.index_select(0, fitted.starts + start)
            )
            & (position < (query.lengths - offset)[:, None])
        ).any(-1)
        offset = offset + _WINDOW
    return differ


def _has_pending_candidates(position: Tensor, *state: Tensor) -> Tensor:
    """Continue while any query has candidates left to examine."""
    return (position >= 0).any()
