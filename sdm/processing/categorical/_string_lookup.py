# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match string categories with compact keys and exact byte comparison."""

from collections.abc import Iterable
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
    """Query hashes and sorted fitted hashes for candidate lookup.

    ``fitted_order`` maps sorted-hash positions back to fitted labels.
    """

    query_hashes: Tensor
    sorted_hashes: Tensor
    fitted_order: Tensor


def _native_string_lookups(
    left: tuple[StringTensor, ...],
    right: tuple[StringTensor, ...],
    max_length: int,
    dtype: torch.dtype,
) -> tuple[Tensor, ...]:
    """Map each query vocabulary to IDs in its fitted vocabulary.

    Missing labels map to ``-1``; the last of duplicate fitted labels wins.
    Keys hold the pair index, so all pairs are matched at once. Keys also
    hold the label length, so a query longer than every fitted label
    matches none: keys are sized from ``max_length``, at least the longest
    fitted label, and compiled graphs do not depend on query lengths.
    """
    # Keys fit the longest fitted label, up to _MAX_KEY_WORDS; longer labels
    # also compare byte windows.
    key_words = min(_MAX_KEY_WORDS, max(1, (max_length + 7) // 8))
    windows = 0
    if max_length > 8 * key_words:
        windows = min(_MAX_WINDOWS, (max_length + _WINDOW - 1) // _WINDOW)
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


def _max_length(values: Iterable[StringTensor]) -> int:
    """Return the byte length of the longest visible label, or zero."""
    longest = 0
    for value in values:
        if value.numel():
            start, end = _label_offsets(value)
            longest = max(longest, int((end - start).max()))
    return longest


def _join_rows(
    values: tuple[StringTensor, ...], index_dtype: torch.dtype
) -> _StringRows:
    """Concatenate the visible labels of one-dimensional string tensors.

    Padding lets byte windows start at any label.
    """
    # Each tensor's offsets index its own bytes; shift them past earlier ones.
    # Narrowing each tensor's offsets keeps wide copies out of compiled code.
    starts, lengths, base = [], [], 0
    for value in values:
        start, end = _label_offsets(value)
        starts.append(start.to(index_dtype) + base)
        lengths.append((end - start).to(index_dtype))
        base += value._data.numel()
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
    return _StringRows(
        data=torch.cat(
            [
                *(value._data for value in values),
                values[0]._data.new_zeros(_WINDOW),
            ]
        ),
        starts=torch.cat(starts),
        lengths=torch.cat(lengths),
        groups=groups,
    )


def _select_rows(rows: _StringRows, index: Tensor | slice) -> _StringRows:
    """Select labels of joined rows, which keep their byte storage."""
    return rows._replace(
        starts=rows.starts[index],
        lengths=rows.lengths[index],
        groups=rows.groups[index],
    )


def _match_string_keys(
    rows: _StringRows,
    query_count: int,
    key_words: int,
    windows: int,
) -> Tensor:
    """Return the fitted row of each query label, or ``-1``.

    The first ``query_count`` rows are query labels, the others fitted
    labels. Only hashes are kept; candidates compare key bytes. Labels
    longer than their keys compare ``windows`` byte windows per step; zero
    means every fitted label fits its key.
    """
    weights = torch.arange(
        start=1,
        end=2 * key_words + 4,
        step=2,
        dtype=torch.int64,
        device=rows.data.device,
    )
    # Hashes filter candidates; key bytes establish equality.
    hashes = _hash_string_keys(rows, key_words, weights)
    sorted_hashes, fitted_order = hashes[query_count:].sort(stable=True)
    keys = _LookupKeys(
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


def _hash_string_keys(
    rows: _StringRows, key_words: int, weights: Tensor
) -> Tensor:
    """Hash label keys, one part of their words at a time.

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
    )


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
    matched = (
        pending
        & (query.lengths == candidates.lengths)
        & (query.groups == candidates.groups)
    )
    # Equal lengths and key bytes prove equality of labels that fit keys.
    for first, last in _key_parts(key_words):
        matched &= (
            _key_bytes(query, key_words, first, last)
            == _key_bytes(candidates, key_words, first, last)
        ).all(-1)
    resolved = -2 - position
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
        resolved = position.where(comparing, resolved)
    position = torch.where(
        matched,
        resolved,
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
