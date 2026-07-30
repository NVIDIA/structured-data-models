from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from typing_extensions import Self, override

from sdm._warnings import warn_once
from sdm.tensor import VarLenTensor
from sdm.tensor.io import arrow_as_tensor

if TYPE_CHECKING:
    import cudf

aten = torch.ops.aten


class StringTensor(VarLenTensor):
    r"""A :class:`torch.Tensor` for UTF-8 encoded string values.

    Args:
        data: Flat contiguous ``uint8`` tensor containing all string values.
        offset: One-dimensional offsets into ``data``.
        size: The shape of the tensor.
        stride: The stride of the tensor.
        storage_offset: The offset into the logical ``offset`` storage.
    """

    # NOTE Assume that `data` stores valid UTF-8 bytes and do not validate it.
    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = (torch.uint8,)

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

        .. code-block:: python

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
            if array.num_chunks == 1:
                array = array.chunk(0)
            else:
                array = array.combine_chunks()

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

        if array.null_count > 0:
            raise ValueError(f"{cls.__name__!r} cannot represent null values")

        buffers = array.buffers()

        return cls(
            data=torch.frombuffer(buffers[2], dtype=torch.uint8).to(device)
            if buffers[2] is not None and buffers[2].size > 0
            else torch.empty(0, dtype=torch.uint8, device=device),
            offset=torch.frombuffer(
                buffer=buffers[1],
                dtype=torch.int32 if is_string else torch.int64,
            ).to(device),
            size=size,
            storage_offset=array.offset,
        )

    @override
    def to_arrow(self) -> pa.Array:
        r"""Convert this tensor to a flat :class:`pyarrow.Array`."""
        tensor = cast(StringTensor, self.contiguous().cpu())

        return pa.Array.from_buffers(
            pa.string()
            if tensor._offset.dtype == torch.int32
            else pa.large_string(),
            length=tensor.numel(),
            buffers=[
                None,
                pa.py_buffer(tensor._offset.numpy()),
                pa.py_buffer(tensor._data.numpy()),
            ],
            offset=int(tensor.storage_offset()),
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

            # StringTensor stores variable-width strings in separate UTF-8
            # data and offset buffers. Use pylibcudf to expose them without a
            # host copy.
            offset_column = plc.Column.from_array(obj=tensor._offset)
            plc_column = plc.Column(
                data_type=plc.DataType(plc.TypeId.STRING),
                size=tensor.numel(),
                data=plc.gpumemoryview(tensor._data),
                mask=None,
                null_count=0,
                offset=int(tensor.storage_offset()),
                children=[offset_column],
            )
            return cudf.Series.from_pylibcudf(plc_column)

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
        if column.null_count() > 0:
            raise ValueError(f"{cls.__name__!r} cannot represent null values")

        # `None` or zero-length when the column holds no characters:
        chars = column.data()
        data = torch.from_dlpack(
            cp.asarray(chars) if chars is not None else cp.empty(0, cp.uint8)
        ).to(device)

        if len(ser) == 0:
            return cls(
                data=data,
                offset=torch.zeros(1, dtype=torch.int32, device=data.device),
                size=size,
            )

        offsets = column.children()[0]  # (base_size + 1,) INT32/INT64 values
        offset_dtype = (
            cp.int32
            if offsets.type().id() == plc.types.TypeId.INT32
            else cp.int64
        )
        return cls(
            data=data,
            offset=torch.from_dlpack(
                cp.asarray(offsets.data()).view(offset_dtype)
            ).to(device),
            size=size,
            storage_offset=column.offset(),
        )

    @classmethod
    @override
    def from_list(
        cls,
        values: str | Sequence[Any],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> Self:
        r"""Create tensor from a rectangular Python list of strings.

        .. code-block:: python

            from sdm import StringTensor

            tensor = VarLenTensor.from_list([
                ["foo", "bar"],
                ["hello world", ""],
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

        def flatten(seq: Any) -> tuple[int, ...]:
            if isinstance(seq, str):
                array.append(seq)
                return ()
            if not isinstance(seq, Sequence):
                raise TypeError(f"{cls.__name__!r} data must contain strings")
            if len(seq) == 0:
                return (0,)

            if not isinstance(seq[0], Sequence) or isinstance(seq[0], str):
                array.extend(seq)
                return (len(seq),)

            child_size: tuple[int, ...] | None = None
            for item in seq:
                item_size = flatten(item)
                if child_size is None:
                    child_size = item_size
                elif item_size != child_size:
                    raise ValueError(
                        f"{cls.__name__!r} data must be rectangular"
                    )

            assert child_size is not None
            return (len(seq), *child_size)

        if isinstance(values, str):
            array: list[str] = [values]
            size: tuple[int, ...] = ()
        else:
            array = []
            size = flatten(values)

        pa_type = pa.large_string()
        if offset_dtype == torch.int32:
            pa_type = pa.string()

        return cls.from_arrow(
            array=pa.array(array, type=pa_type),
            device=device,
            size=size,
        )

    @override
    def item(self) -> str:  # type: ignore
        return cast(str, super().item())

    def __eq__(self, other: object) -> Tensor:  # type: ignore
        if isinstance(other, str):
            return _eq(self, other)
        return cast(Tensor, super().__eq__(other))

    def __ne__(self, other: object) -> Tensor:  # type: ignore
        if isinstance(other, str):
            return _ne(self, other)
        return cast(Tensor, super().__ne__(other))

    def __str__(self) -> str:
        return self.item() if self.numel() == 1 else self.__repr__()

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}(..."
        out += f", size={tuple(self.size())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        out += ")"
        return out


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
        mask = arrow_as_tensor(out, dtype=torch.bool, device=inp.device)
    else:
        assert backend == "cudf"
        with torch.cuda.device(inp.device):
            if isinstance(other, StringTensor):
                out = inp.to_cudf() == other.to_cudf()
            else:
                out = inp.to_cudf() == other
            mask = torch.from_dlpack(out.to_cupy()).view(size)

    return mask.view(size)


@StringTensor.implements(aten.ne.Tensor)
@StringTensor.implements(aten.ne.str)
def _ne(inp: StringTensor, other: Tensor | str) -> Tensor:
    return ~_eq(inp, other)


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
