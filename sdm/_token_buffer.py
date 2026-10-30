# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Literal, Self, cast

import torch
from torch import Tensor

from sdm._memory import chunk_memory_limit, set_pinned_max_round_threshold
from sdm.cache import Cache, KVCacheEntry

# Lets one chunk load, another compute, and a third store concurrently.
_NUM_SLOTS = 3


class TokenBuffer:
    r"""A buffer of ``T`` tokens per row for row embeddings.

    Row embeddings update the buffer in passes: :meth:`rows` updates rows
    independently and :meth:`columns` updates token columns independently.

    The buffer is stored column-major as ``[..., T, R, D]``. Column passes
    see slices of it, and row passes see ``[..., R, T, D]`` views.

    Without offloading, the buffer is a device tensor and each pass computes
    on the whole buffer at once. With ``offload="cpu"``, the buffer lives in
    pinned host memory, and each pass streams chunks of it through a few
    device slots. Host-to-device copies, compute and device-to-host copies run
    on separate streams, so copies of neighboring chunks overlap with compute.

    Args:
        size: The buffer size ``[..., R, T, D]``. Offloading requires leading
            dimensions of size one, *i.e.*, a single estimator and no class
            decomposition.
        dtype: The buffer dtype.
        device: The compute device.
        offload: Where to keep the buffer: ``None`` keeps it on ``device`` and
            ``"cpu"`` keeps it in pinned host memory.
    """

    def __init__(
        self,
        size: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
        offload: Literal["cpu"] | None = None,
    ) -> None:
        self.size = torch.Size(size)
        self.dtype = dtype
        self.device = device
        self.offload = offload
        *B, R, T, D = self.size

        if offload is None:
            self.tensor = torch.empty(
                (*B, T, R, D), dtype=dtype, device=device
            )
            return

        if math.prod(B) != 1:
            raise ValueError(
                "'offload' requires 'estimator_batch_size=1' and no more "
                "target classes than the model predicts natively (got "
                f"buffer size {self.size})"
            )
        self._pitch = R * D * dtype.itemsize
        set_pinned_max_round_threshold(mb=1)
        self._host = torch.empty(
            self.size.numel(), dtype=dtype, pin_memory=True
        )
        slot_bytes = max(
            chunk_memory_limit(device) // 2,
            T * D * dtype.itemsize,
            self._pitch,
        )
        self._slots = [
            torch.empty(slot_bytes, dtype=torch.uint8, device=device)
            for _ in range(_NUM_SLOTS)
        ]
        self._h2d = torch.cuda.Stream(device)
        self._d2h = torch.cuda.Stream(device)
        self._free = [torch.cuda.Event() for _ in self._slots]

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self:
        r"""Wraps a column-major ``[..., T, R, D]`` device tensor."""
        buffer = cls.__new__(cls)
        *B, T, R, D = tensor.size()
        buffer.size = torch.Size((*B, R, T, D))
        buffer.dtype = tensor.dtype
        buffer.device = tensor.device
        buffer.offload = None
        buffer.tensor = tensor
        return buffer

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.offload is None:
            return
        # The pinned allocator does not track these copies, so the host table
        # may only be freed or reused once they complete.
        self._h2d.synchronize()
        self._d2h.synchronize()

    def readout(self, num_tokens: int) -> Tensor:
        r"""Device tensor for the first ``num_tokens`` tokens of every row.

        Without offloading, this is a view of the buffer. Otherwise, it is a
        separate ``[..., R, num_tokens, D]`` tensor for the last row pass to
        write into.
        """
        if self.offload is None:
            return self.tensor[..., :num_tokens, :, :].transpose(-2, -3)
        return torch.empty(
            (*self.size[:-2], num_tokens, self.size[-1]),
            dtype=self.dtype,
            device=self.device,
        )

    def rows(
        self,
        compute: Callable[[Tensor, slice], object],
        *,
        load: bool = True,
        store: bool = True,
    ) -> None:
        r"""Updates rows in place.

        Args:
            compute: Called as ``compute(tokens, rows)`` with ``tokens`` of
                shape ``[..., N, T, D]`` holding the buffer rows ``rows``.
            load: Whether ``compute`` reads ``tokens``. If ``False``,
                ``compute`` must fill them.
            store: Whether the updated rows are needed afterwards.
        """
        if self.offload is None:
            compute(self.tensor.transpose(-2, -3), slice(0, self.size[-3]))
            return

        *B, R, T, D = self.size
        token_bytes = D * self.dtype.itemsize
        step = max(1, len(self._slots[0]) // (T * token_bytes))
        chunks = [
            slice(start, min(start + step, R)) for start in range(0, R, step)
        ]
        self._stream(
            chunks=[
                (
                    chunk.start * token_bytes,
                    (chunk.stop - chunk.start) * token_bytes,
                    T,
                )
                for chunk in chunks
            ],
            compute=lambda slot, k: compute(
                slot.view(self.dtype)
                .view(*B, T, chunks[k].stop - chunks[k].start, D)
                .transpose(-2, -3),
                chunks[k],
            ),
            load=load,
            store=store,
        )

    def columns(
        self,
        compute: Callable[[Tensor], object],
        tokens: slice,
    ) -> None:
        r"""Updates the token columns ``tokens`` in place.

        Args:
            compute: Called on chunks of consecutive token columns, in order,
                with ``cells`` of shape ``[..., N, R, D]``.
            tokens: The token columns to update.
        """
        if self.offload is None:
            compute(self.tensor[..., tokens, :, :])
            return

        *B, R, _, D = self.size
        step = max(1, len(self._slots[0]) // self._pitch)
        chunks = [
            slice(start, min(start + step, tokens.stop))
            for start in range(tokens.start, tokens.stop, step)
        ]
        self._stream(
            chunks=[
                (
                    chunk.start * self._pitch,
                    self._pitch,
                    chunk.stop - chunk.start,
                )
                for chunk in chunks
            ],
            compute=lambda slot, k: compute(
                slot.view(self.dtype).view(
                    *B, chunks[k].stop - chunks[k].start, R, D
                )
            ),
        )

    def _stream(
        self,
        chunks: list[tuple[int, int, int]],
        compute: Callable[[Tensor, int], object],
        *,
        load: bool = True,
        store: bool = True,
    ) -> None:
        # A chunk holds `height` host rows of `width` bytes at byte `offset`,
        # stored densely in its slot.
        from cuda.bindings import runtime

        def copy(
            dst: int, dpitch: int, src: int, spitch: int, chunk, kind, stream
        ) -> None:
            _, width, height = chunk
            (error,) = runtime.cudaMemcpy2DAsync(
                dst,
                dpitch,
                src,
                spitch,
                width,
                height,
                kind,
                stream.cuda_stream,
            )
            if error != runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Host offloading copy failed ({error})")

        host = self._host.data_ptr()
        compute_stream = torch.cuda.current_stream(self.device)
        loaded = [torch.cuda.Event() for _ in self._slots]
        # Stores of the previous pass must land before this pass loads.
        self._h2d.wait_stream(self._d2h)

        def issue_load(k: int) -> None:
            i = k % len(self._slots)
            self._h2d.wait_event(self._free[i])
            copy(
                dst=self._slots[i].data_ptr(),
                dpitch=chunks[k][1],
                src=host + chunks[k][0],
                spitch=self._pitch,
                chunk=chunks[k],
                kind=runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream=self._h2d,
            )
            loaded[i].record(self._h2d)

        if load and chunks:
            issue_load(0)
        for k, (offset, width, height) in enumerate(chunks):
            i = k % len(self._slots)
            if load and k + 1 < len(chunks):
                issue_load(k + 1)
            compute_stream.wait_event(loaded[i] if load else self._free[i])
            compute(self._slots[i][: width * height], k)
            if not store:
                self._free[i].record(compute_stream)
                continue
            self._d2h.wait_stream(compute_stream)
            copy(
                dst=host + offset,
                dpitch=self._pitch,
                src=self._slots[i].data_ptr(),
                spitch=width,
                chunk=chunks[k],
                kind=runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream=self._d2h,
            )
            self._free[i].record(self._d2h)


def run_column_layers(
    buffer: TokenBuffer,
    layers: Sequence[tuple[str, torch.nn.Module]],
    *,
    tokens: slice,
    key_rows: Sequence[Any],
    cache: Cache | None,
    epilogue: Callable[[Tensor], object] | None = None,
) -> None:
    r"""Runs consecutive column layers over ``buffer`` in a single pass.

    Records or replays each layer's key/value projections under its cache key.

    Args:
        buffer: The token buffer.
        layers: Pairs of cache entry name and column layer, in order.
        tokens: The token columns to update.
        key_rows: Per layer, the row index of context cells to attend to.
        cache: The cache to record into or replay from, or ``None``.
        epilogue: Called on each chunk of cells after the last layer.
    """
    recording = cache is not None and cache.is_recording
    replaying = cache is not None and cache.is_replaying
    # Replayed entries span all token columns, so they need a single chunk.
    assert not replaying or buffer.offload is None
    entries: dict[str, list[KVCacheEntry]] = {key: [] for key, _ in layers}

    def compute(cells: Tensor) -> None:  # [..., N, R, D]
        for (key, layer), rows in zip(layers, key_rows, strict=True):
            result = layer(
                query=cells,
                key_value=cast(KVCacheEntry, cast(Cache, cache)[key])
                if replaying
                else cells[..., rows, :],
                return_key_value=recording,
                batch_size_limit="auto",
                out=cells,
            )
            if recording:
                entries[key].append(result[1])
            del result
        if epilogue is not None:
            epilogue(cells)

    buffer.columns(compute, tokens)

    if recording:
        assert cache is not None
        dim = len(buffer.size) - 3
        for key, parts in entries.items():
            cache[key] = (
                parts[0]
                if len(parts) == 1
                else KVCacheEntry(
                    key=torch.cat([part.key for part in parts], dim=dim),
                    value=torch.cat([part.value for part in parts], dim=dim),
                )
            )


def add_labels(
    cells: Tensor,  # [..., N, C, D]
    y_emb: Tensor,  # [..., R_train, D]
    rows: slice,
) -> None:
    r"""Adds label embeddings to the context rows among ``rows``."""
    stop = min(rows.stop, y_emb.size(-2))
    if stop > rows.start:
        cells[..., : stop - rows.start, :, :] += y_emb[
            ..., rows.start : stop, None, :
        ].to(cells.dtype)
