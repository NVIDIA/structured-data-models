from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from typing_extensions import override

from sdm._warnings import warn_once
from sdm.tensor import VarLenTensor
from sdm.tensor.io import arrow_as_tensor
from sdm.tensor.io.arrow import _combine_arrow_chunks

if TYPE_CHECKING:
    import cudf

aten = torch.ops.aten

_STRING_HASH_BASE = 257
_STRING_HASH_LENGTH_MIX = -7046029288634856825
_STRING_MATCH_BYTE_LANES = 64
_STRING_MATCH_ROW_LANES = 1024


class StringTensor(VarLenTensor):
    r"""A :class:`torch.Tensor` for UTF-8 encoded string values.

    Args:
        data: Flat contiguous ``uint8`` tensor containing all string values.
        offset: One-dimensional offsets into ``data``.
        valid: One-dimensional mask indicating valid, non-null string values.
        size: The shape of the tensor.
        stride: The stride of the tensor.
        storage_offset: The offset into the logical ``offset`` storage.
    """

    # NOTE Assume that `data` stores valid UTF-8 bytes and do not validate it.
    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = (torch.uint8,)
    __hash__ = Tensor.__hash__

    @override
    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a string :class:`pyarrow.Array`.

        .. testcode::

            import pyarrow as pa
            from sdm import StringTensor

            array = pa.array(["foo", "bar", "hello world", ""])
            tensor = StringTensor.from_arrow(array, size=(2, 2))

        Args:
            array: The string :class:`pyarrow.Array` or
                :class:`pyarrow.ChunkedArray`.
            size: The shape of the tensor.
            device: The device.
        """
        if isinstance(array, pa.ChunkedArray):
            array = _combine_arrow_chunks(array)

        if size is None:
            size = (len(array),)
        elif math.prod(size) != len(array):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_arrow' to contain "
                f"{len(array)} elements (got {math.prod(size)})"
            )

        is_string = pa.types.is_string(array.type)
        is_large_string = pa.types.is_large_string(array.type)
        if not is_string and not is_large_string:
            raise TypeError(
                f"Expected 'array' in '{cls.__name__}.from_arrow' to have "
                f"'string' or 'large_string' type (got '{array.type}')"
            )

        buffers = array.buffers()
        offset = torch.frombuffer(
            buffers[1],
            dtype=torch.int32 if is_string else torch.int64,
        )
        valid: Tensor | None = None
        storage_offset = array.offset
        if array.null_count > 0:
            offset = offset[array.offset : array.offset + len(array) + 1]
            valid = arrow_as_tensor(
                array.is_valid(),
                dtype=torch.bool,
                device=device,
            )
            storage_offset = 0

        return cls(
            data=torch.frombuffer(buffers[2], dtype=torch.uint8).to(device)
            if buffers[2] is not None and buffers[2].size > 0
            else torch.empty(0, dtype=torch.uint8, device=device),
            offset=offset.to(device),
            valid=valid,
            size=size,
            storage_offset=storage_offset,
        )

    @override
    def to_arrow(self) -> pa.Array:
        r"""Convert this tensor to a flat :class:`pyarrow.Array`."""
        tensor = cast(StringTensor, self.contiguous().cpu())
        if tensor.numel() == 0 and tensor.storage_offset() != 0:
            tensor = cast(StringTensor, tensor.clone())

        return pa.Array.from_buffers(
            pa.string()
            if tensor._offset.dtype == torch.int32
            else pa.large_string(),
            length=tensor.numel(),
            buffers=[
                pa.array(tensor._valid.numpy(), type=pa.bool_()).buffers()[1]
                if tensor._valid is not None
                else None,
                pa.py_buffer(tensor._offset.numpy()),
                pa.py_buffer(tensor._data.numpy()),
            ],
            null_count=-1,
            offset=int(tensor.storage_offset()),
        )

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series | cudf.Index,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a string :class:`cudf.Series`.

        Args:
            ser: The string :class:`cudf.Series` or :class:`cudf.Index`.
            size: The shape of the tensor.
            device: The device.
        """
        import cupy as cp
        import pylibcudf as plc
        from cudf.api.types import is_string_dtype

        if size is None:
            size = (len(ser),)
        elif math.prod(size) != len(ser):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_cudf' to contain "
                f"{len(ser)} elements (got {math.prod(size)})"
            )

        if not is_string_dtype(ser.dtype):
            raise TypeError(
                f"Expected 'values' in '{cls.__name__}.from_cudf' to have "
                f"string type (got '{ser.dtype}')"
            )

        # `Series.to_pylibcudf` returns a zero-copy Arrow-style view: base
        # character/offset buffers plus a row offset into the offsets.
        column, _ = ser.to_pylibcudf()

        # `None` or zero-length when the column holds no characters:
        chars = column.data()
        data = torch.from_dlpack(
            cp.asarray(chars) if chars is not None else cp.empty(0, cp.uint8)
        ).to(device)

        if len(ser) == 0:
            return cls(
                data=data,
                offset=torch.zeros(1, dtype=torch.int32, device=data.device),
                valid=None,
                size=size,
            )

        offsets = column.children()[0]  # (base_size + 1,) INT32/INT64 values
        offset_dtype = (
            cp.int32
            if offsets.type().id() == plc.types.TypeId.INT32
            else cp.int64
        )
        offset = torch.from_dlpack(
            cp.asarray(offsets.data()).view(offset_dtype)
        )

        valid: Tensor | None = None
        storage_offset = column.offset()
        if ser.hasnans:
            offset = offset[column.offset() : column.offset() + len(ser) + 1]
            valid = torch.from_dlpack(ser.notnull().to_cupy()).to(device)
            storage_offset = 0

        return cls(
            data=data,
            offset=offset.to(device),
            valid=valid,
            size=size,
            storage_offset=storage_offset,
        )

    def to_cudf(self) -> cudf.Series:
        r"""Convert this CUDA tensor to a flat :class:`cudf.Series`."""
        if not self.is_cuda:
            raise RuntimeError(
                f"Expected tensor to be on a CUDA device (got '{self.device}')"
            )

        tensor = cast(StringTensor, self.contiguous())

        with torch.cuda.device(self.device):
            import cudf
            import pylibcudf as plc

            mask = None
            null_count = 0
            start = int(tensor.storage_offset())
            if tensor._valid is not None:
                mask = cudf.Series(tensor._valid, copy=False)._column.as_mask()
                if isinstance(mask, tuple):
                    mask = mask[0]
                null_count = plc.null_mask.null_count(
                    mask, start, start + tensor.numel()
                )

            plc_column = plc.Column(
                data_type=plc.DataType(plc.TypeId.STRING),
                size=tensor.numel(),
                data=plc.gpumemoryview(tensor._data),
                mask=mask,
                null_count=null_count,
                offset=start,
                children=[plc.Column.from_array(obj=tensor._offset)],
            )
            return cudf.Series.from_pylibcudf(plc_column)

    @classmethod
    @override
    def from_list(
        cls,
        values: str | Sequence[Any] | None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> Self:
        r"""Create tensor from a rectangular Python list of strings.

        .. testcode::

            from sdm import StringTensor

            tensor = StringTensor.from_list([
                ["foo", "bar"],
                ["hello world", ""],
                [None, "xzy"],
            ])

        Args:
            values: The rectangular Python list of strings.
            dtype: The dtype of the ``value`` tensor.
            device: The device.
            offset_dtype: The dtype of the ``offset`` tensor.
        """
        if dtype is not None and dtype != torch.uint8:
            raise ValueError(
                f"Expected 'dtype' in '{cls.__name__}.from_list' to be "
                f"'torch.uint8' (got '{dtype}')"
            )
        if offset_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Expected 'offset_dtype' in '{cls.__name__}.from_list' "
                f"to be 'torch.int32' or 'torch.int64' "
                f"(got '{offset_dtype}')"
            )

        data: list[str | None] = []

        def flatten(value: Any) -> tuple[int, ...]:
            if value is None or isinstance(value, str):
                data.append(value)
                return ()

            if not isinstance(value, Sequence):
                raise ValueError(f"{cls.__name__!r} data must be rectangular")

            if len(value) == 0:
                return (0,)

            if not isinstance(value[0], Sequence) or isinstance(value[0], str):
                data.extend(value)
                return (len(value),)

            child_size: tuple[int, ...] | None = None
            for item in value:
                item_size = flatten(item)
                if child_size is None:
                    child_size = item_size
                elif item_size != child_size:
                    raise ValueError(
                        f"{cls.__name__!r} data must be rectangular"
                    )

            assert child_size is not None
            return (len(value), *child_size)

        size = flatten(values)

        pa_type = pa.large_string()
        if offset_dtype == torch.int32:
            pa_type = pa.string()

        return cls.from_arrow(
            array=pa.array(data, type=pa_type),
            device=device,
            size=size,
        )

    @override
    def item(self) -> str | None:  # type: ignore
        return cast(str | None, super().item())

    def __eq__(self, other: object) -> Tensor:  # type: ignore
        if isinstance(other, str):
            return _eq(self, other)
        return cast(Tensor, super().__eq__(other))

    def __ne__(self, other: object) -> Tensor:  # type: ignore
        if isinstance(other, str):
            return _ne(self, other)
        return cast(Tensor, super().__ne__(other))

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}("
        out += f"size={tuple(self.size())}"
        if self.valid is not None:
            out += f", null_count={int((~self.valid).sum())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        out += ")"
        return out


def _validate_find_equal_inputs(
    left: StringTensor,
    right: StringTensor,
) -> None:
    if left.dim() != 1 or right.dim() != 1:
        raise ValueError("Expected one-dimensional string tensors")
    if left.device != right.device:
        raise RuntimeError(
            "Expected both tensors to be on the same device "
            f"(got '{left.device}' and '{right.device}')"
        )


def _string_bounds(
    inp: StringTensor,
) -> tuple[Tensor, Tensor, Tensor]:
    position = torch.arange(
        inp.size(0),
        dtype=torch.int64,
        device=inp.device,
    )
    position = position * inp.stride(0) + inp._storage_offset
    start = inp._offset[position]
    end = inp._offset[position + 1]
    if inp._valid is None:
        valid = torch.ones(inp.size(), dtype=torch.bool, device=inp.device)
    else:
        valid = inp._valid[position]
    return start, end, valid


def _string_metadata(
    inp: StringTensor,
    *,
    active: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    start, end, valid = _string_bounds(inp)
    if active is not None:
        valid = valid & active
    start = start.to(torch.int64)
    length = (end.to(torch.int64) - start).masked_fill(~valid, 0)
    return start, length, valid


def _hash_strings(
    data: Tensor,
    metadata: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    start, length, valid = metadata
    row_lanes = torch.arange(
        min(start.numel(), _STRING_MATCH_ROW_LANES),
        dtype=torch.int64,
        device=data.device,
    )
    byte_lanes = torch.arange(
        _STRING_MATCH_BYTE_LANES,
        dtype=torch.int64,
        device=data.device,
    )
    minimum = torch.iinfo(torch.int64).min
    output_size = length.numel()
    if output_size > _STRING_MATCH_ROW_LANES:
        output_size = (
            (output_size + _STRING_MATCH_ROW_LANES - 1)
            // _STRING_MATCH_ROW_LANES
            * _STRING_MATCH_ROW_LANES
        )
    output = length.new_full((output_size,), minimum)
    row_offset = row_lanes.new_zeros(())

    def row_cond(row_offset: Tensor, output: Tensor) -> Tensor:
        return row_offset < start.numel()

    def row_body(
        row_offset: Tensor,
        output: Tensor,
    ) -> tuple[Tensor, Tensor]:
        unbounded_rows = row_offset + row_lanes
        row_valid = unbounded_rows < start.numel()
        rows = unbounded_rows.clamp(max=start.numel() - 1)
        row_start = start[rows]
        row_length = length[rows]
        row_active = valid[rows] & row_valid
        position = byte_lanes.new_zeros(())
        row_hash = torch.zeros_like(row_length)

        def byte_cond(position: Tensor, row_hash: Tensor) -> Tensor:
            return ((position < row_length) & row_active).any()

        def byte_body(
            position: Tensor,
            row_hash: Tensor,
        ) -> tuple[Tensor, Tensor]:
            byte_position = position + byte_lanes
            byte_valid = (
                byte_position < row_length.unsqueeze(1)
            ) & row_active.unsqueeze(1)
            data_index = (row_start.unsqueeze(1) + byte_position).clamp(
                max=data.numel() - 1
            )
            byte = (data[data_index].to(torch.int64) + 1).masked_fill(
                ~byte_valid,
                0,
            )
            segment_length = (row_length - position).clamp(
                min=0,
                max=_STRING_MATCH_BYTE_LANES,
            )
            exponent = (
                segment_length.unsqueeze(1) - byte_lanes - 1
            ).clamp_min(0)
            base = torch.full_like(byte_lanes, _STRING_HASH_BASE)
            segment_hash = (byte * base.pow(exponent)).sum(dim=1)
            row_hash = (
                row_hash
                * torch.full_like(segment_length, _STRING_HASH_BASE).pow(
                    segment_length
                )
                + segment_hash
            )
            return position + _STRING_MATCH_BYTE_LANES, row_hash

        row_hash = torch.while_loop(
            byte_cond,
            byte_body,
            (position, row_hash),
        )[1]
        row_hash = (
            row_hash + row_length * _STRING_HASH_LENGTH_MIX
        ).masked_fill(~row_active, 0)
        # One complete lane block is already in output order.
        if start.numel() <= _STRING_MATCH_ROW_LANES:
            output = row_hash
        else:
            output = output.index_copy(
                0,
                unbounded_rows,
                row_hash.masked_fill(~row_valid, minimum),
            )
        return row_offset + row_lanes.numel(), output

    return torch.while_loop(
        row_cond,
        row_body,
        (row_offset, output),
    )[1][: length.numel()]


def _match_hash_candidates(
    left_data: Tensor,
    right_data: Tensor,
    permutation: Tensor,
    lower: Tensor,
    upper: Tensor,
    left: tuple[Tensor, Tensor, Tensor],
    right: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    left_start, left_length, left_valid = left
    right_start, right_length, right_valid = right
    # Higher-order operators reject aliased closure inputs. Metadata is small
    # compared with the byte buffers, so make the candidate side independent.
    right_start = right_start.clone()
    right_length = right_length.clone()
    right_valid = right_valid.clone()
    row_lanes = torch.arange(
        min(left_start.numel(), _STRING_MATCH_ROW_LANES),
        dtype=torch.int64,
        device=left_data.device,
    )
    byte_lanes = torch.arange(
        _STRING_MATCH_BYTE_LANES,
        dtype=torch.int64,
        device=left_data.device,
    )
    output_size = lower.numel()
    if output_size > _STRING_MATCH_ROW_LANES:
        output_size = (
            (output_size + _STRING_MATCH_ROW_LANES - 1)
            // _STRING_MATCH_ROW_LANES
            * _STRING_MATCH_ROW_LANES
        )
    output = lower.new_full((output_size,), -1)
    row_offset = row_lanes.new_zeros(())

    def row_cond(row_offset: Tensor, output: Tensor) -> Tensor:
        return row_offset < left_start.numel()

    def row_body(
        row_offset: Tensor,
        output: Tensor,
    ) -> tuple[Tensor, Tensor]:
        unbounded_rows = row_offset + row_lanes
        row_valid = unbounded_rows < left_start.numel()
        rows = unbounded_rows.clamp(max=left_start.numel() - 1)
        row_start = left_start[rows]
        row_length = left_length[rows]
        row_active = left_valid[rows] & row_valid
        row_lower = lower[rows]
        row_upper = upper[rows]

        def bounded_candidate_match(
            candidate: Tensor,
            pending: Tensor,
        ) -> Tensor:
            match = (
                pending
                & right_valid[candidate]
                & (row_length == right_length[candidate])
            )
            position = byte_lanes.new_zeros(())

            def compare_cond(position: Tensor, match: Tensor) -> Tensor:
                return ((position < row_length) & match).any()

            def compare_body(
                position: Tensor,
                match: Tensor,
            ) -> tuple[Tensor, Tensor]:
                byte_position = position + byte_lanes
                byte_valid = (
                    byte_position < row_length.unsqueeze(1)
                ) & match.unsqueeze(1)
                left_index = row_start.unsqueeze(1) + byte_position
                right_index = (
                    right_start[candidate].unsqueeze(1) + byte_position
                )
                left_index = left_index.clamp(max=left_data.numel() - 1)
                right_index = right_index.clamp(max=right_data.numel() - 1)
                if left_data is right_data:
                    byte_equal = (
                        left_data[left_index] == left_data[right_index]
                    )
                else:
                    byte_equal = (
                        left_data[left_index] == right_data[right_index]
                    )
                byte_match = torch.where(byte_valid, byte_equal, True)
                match = match & byte_match.all(dim=1)
                return position + _STRING_MATCH_BYTE_LANES, match

            return torch.while_loop(
                compare_cond,
                compare_body,
                (position, match),
            )[1]

        candidate_offset = row_lower.new_zeros(())
        pending = (row_lower < row_upper) & row_active
        first_candidate = permutation[
            row_lower.clamp(max=permutation.numel() - 1)
        ]
        # Most buckets contain one value; duplicate buckets match their stable
        # first candidate without materializing the bucket's Cartesian product.
        first_match = bounded_candidate_match(first_candidate, pending)
        row_output = torch.where(first_match, first_candidate, -1)
        candidate_offset = candidate_offset + 1

        def candidate_cond(
            candidate_offset: Tensor,
            row_output: Tensor,
        ) -> Tensor:
            return (
                (row_lower + candidate_offset < row_upper)
                & (row_output < 0)
                & row_active
            ).any()

        def candidate_body(
            candidate_offset: Tensor,
            row_output: Tensor,
        ) -> tuple[Tensor, Tensor]:
            pending = (
                (row_lower + candidate_offset < row_upper)
                & (row_output < 0)
                & row_active
            )
            candidate_position = (row_lower + candidate_offset).clamp(
                max=permutation.numel() - 1
            )
            candidate = permutation[candidate_position]
            match = bounded_candidate_match(candidate, pending)
            row_output = torch.where(match, candidate, row_output)
            return candidate_offset + 1, row_output

        row_output = torch.while_loop(
            candidate_cond,
            candidate_body,
            (candidate_offset, row_output),
        )[1]
        # One complete lane block is already in output order.
        if left_start.numel() <= _STRING_MATCH_ROW_LANES:
            output = row_output
        else:
            output = output.index_copy(
                0,
                unbounded_rows,
                row_output.masked_fill(~row_valid, -1),
            )
        return row_offset + row_lanes.numel(), output

    return torch.while_loop(
        row_cond,
        row_body,
        (row_offset, output),
    )[1][: left_start.numel()]


def _find_equal_indices(
    left: StringTensor,
    right: StringTensor,
    *,
    left_active: Tensor | None = None,
) -> Tensor:
    _validate_find_equal_inputs(left, right)
    num_left = left.numel()
    num_right = right.numel()
    if num_left == 0 or num_right == 0:
        return torch.full(
            (num_left,),
            -1,
            dtype=torch.int64,
            device=left.device,
        )

    # Inductor lowers loop bodies even when all string lengths are zero, so
    # keep their otherwise-empty byte buffers safe to index.
    same_data = left._data is right._data
    left_data = left._data
    if left_data.numel() == 0:
        left_data = left_data.new_zeros(1)
    if same_data:
        right_data = left_data
    else:
        right_data = right._data
        if right_data.numel() == 0:
            right_data = right_data.new_zeros(1)

    left_metadata = _string_metadata(left, active=left_active)
    right_metadata = _string_metadata(right)
    # Hashing narrows candidate ranges; matches are still verified bytewise.
    left_hash = _hash_strings(left_data, left_metadata)
    right_hash = _hash_strings(right_data, right_metadata)
    sorted_hash, permutation = right_hash.sort(stable=True)
    lower = torch.searchsorted(sorted_hash, left_hash)
    upper = torch.searchsorted(sorted_hash, left_hash, right=True)
    return _match_hash_candidates(
        left_data,
        right_data,
        permutation,
        lower,
        upper,
        left_metadata,
        right_metadata,
    )


@StringTensor.implements(aten.eq.Tensor)
@StringTensor.implements(aten.eq.str)
def _eq(inp: StringTensor, other: Tensor | str) -> Tensor:
    if isinstance(other, Tensor) and inp.device != other.device:
        raise RuntimeError(
            f"Expected both tensors to be on the same device "
            f"(got '{inp.device}' and '{other.device}')"
        )

    if not isinstance(other, StringTensor | str):
        return torch.zeros(
            torch.broadcast_shapes(inp.size(), other.size()),
            dtype=torch.bool,
            device=inp.device,
        )

    if isinstance(other, Tensor):
        size = torch.broadcast_shapes(inp.size(), other.size())
        inp = cast(StringTensor, inp.expand(size))
        other = cast(StringTensor, other.expand(size))
    else:
        size = inp.size()

    backend: Literal["arrow", "cudf"] = "arrow"
    if inp.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-eq",
                message=(
                    "Falling back to a CPU-based string comparison because "
                    "cuDF is not installed. Install cuDF to enable faster "
                    "CUDA-based string comparisons without device "
                    "synchronization."
                ),
            )

    if backend == "arrow":
        out = pc.call_function(
            "equal",
            [
                inp.to_arrow(),
                other.to_arrow() if isinstance(other, StringTensor) else other,
            ],
        )
        if out.null_count > 0:
            out = out.fill_null(False)
        mask = arrow_as_tensor(out, dtype=torch.bool, device=inp.device)
    else:
        assert backend == "cudf"
        with torch.cuda.device(inp.device):
            if isinstance(other, StringTensor):
                out = inp.to_cudf() == other.to_cudf()
            else:
                out = inp.to_cudf() == other
            if out.hasnans:
                out = out.fillna(False)
            mask = torch.from_dlpack(out.to_cupy()).view(size)

    return mask.view(size)


@StringTensor.implements(aten.ne.Tensor)
@StringTensor.implements(aten.ne.str)
def _ne(inp: StringTensor, other: Tensor | str) -> Tensor:
    if isinstance(other, Tensor) and inp.device != other.device:
        raise RuntimeError(
            f"Expected both tensors to be on the same device "
            f"(got '{inp.device}' and '{other.device}')"
        )

    if not isinstance(other, StringTensor | str):
        return torch.ones(
            torch.broadcast_shapes(inp.size(), other.size()),
            dtype=torch.bool,
            device=inp.device,
        )

    if isinstance(other, Tensor):
        size = torch.broadcast_shapes(inp.size(), other.size())
        inp = cast(StringTensor, inp.expand(size))
        other = cast(StringTensor, other.expand(size))
    else:
        size = inp.size()

    backend: Literal["arrow", "cudf"] = "arrow"
    if inp.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-eq",
                message=(
                    "Falling back to a CPU-based string comparison because "
                    "cuDF is not installed. Install cuDF to enable faster "
                    "CUDA-based string comparisons without device "
                    "synchronization."
                ),
            )

    if backend == "arrow":
        out = pc.call_function(
            "not_equal",
            [
                inp.to_arrow(),
                other.to_arrow() if isinstance(other, StringTensor) else other,
            ],
        )
        if out.null_count > 0:
            out = out.fill_null(False)
        mask = arrow_as_tensor(out, dtype=torch.bool, device=inp.device)
    else:
        assert backend == "cudf"
        with torch.cuda.device(inp.device):
            if isinstance(other, StringTensor):
                out = inp.to_cudf() != other.to_cudf()
            else:
                out = inp.to_cudf() != other
            if out.hasnans:
                out = out.fillna(False)
            mask = torch.from_dlpack(out.to_cupy()).view(size)

    return mask.view(size)


@StringTensor.implements(aten.sort.default)
@StringTensor.implements(aten.sort.stable)
def _sort(
    inp: StringTensor,
    dim: int = -1,
    descending: bool = False,
    *,
    stable: bool | None = None,
) -> tuple[StringTensor, Tensor]:
    if dim < -inp.dim() or dim >= inp.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.dim()}, {inp.dim() - 1}], but got {dim})"
        )

    if inp.dim() != 1:
        raise NotImplementedError("'sort' only supports one-dimensional input")

    backend: Literal["arrow", "cudf"] = "arrow"
    if inp.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-sort",
                message=(
                    "Falling back to a CPU-based string sort because cuDF is "
                    "not installed. Install cuDF to enable faster CUDA-based "
                    "string sorting without device synchronization."
                ),
            )

    if backend == "arrow":
        out = pc.call_function(
            "array_sort_indices",
            [inp.to_arrow()],
            options=pc.ArraySortOptions(
                order="descending" if descending else "ascending",
            ),
        )
        perm = arrow_as_tensor(out, dtype=torch.int64, device=inp.device)
    else:
        assert backend == "cudf"
        with torch.cuda.device(inp.device):
            perm_ser = inp.to_cudf().argsort(ascending=not descending)
            perm = torch.from_dlpack(perm_ser.astype("int64").to_cupy())

    return cast(StringTensor, inp[perm]), perm
