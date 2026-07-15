from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import Self, override

from sdm.tensor import VarLenTensor

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
            raise ValueError(f"'{cls.__name__}' cannot represent null values")

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
        r"""Convert this CUDA tensor to a flat :class:`cudf.Series`.

        Raises:
            RuntimeError: If this tensor is not CUDA-resident.
            ImportError: If cuDF is not installed.
        """
        if not self.is_cuda:
            raise RuntimeError(
                f"Expected '{self.__class__.__name__}' in 'to_cudf' to be "
                f"CUDA-resident (got '{self.device}')"
            )

        with torch.cuda.device(self.device):
            try:
                import cudf
                import pylibcudf as plc
            except ImportError as exc:
                raise ImportError(
                    "Converting tensors to cuDF requires cuDF"
                ) from exc

            tensor = self
            tensor = tensor.contiguous()

            # StringTensor stores variable-width strings in separate UTF-8
            # data and offset buffers. Use pylibcudf to expose them without a
            # host copy.
            offset_column = plc.Column.from_array(  # ty: ignore[missing-argument]
                obj=tensor._offset
            )
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

        column = ser._column
        if column.null_count > 0:
            raise ValueError(f"'{cls.__name__}' cannot represent null values")

        if len(ser) == 0:
            data = torch.from_dlpack(cp.asarray(column.data)).to(device)
            return cls(
                data=data,
                offset=torch.zeros(1, dtype=torch.int32, device=data.device),
                size=size,
            )

        return cls(
            data=torch.from_dlpack(cp.asarray(column.data)).to(device),
            offset=torch.from_dlpack(cp.asarray(column.children[0])).to(
                device
            ),
            size=size,
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
                raise TypeError(f"'{cls.__name__}' data must contain strings")
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
                        f"'{cls.__name__}' data must be rectangular"
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

    import pyarrow.compute as pc

    out = pc.call_function(  # TODO Add GPU implementation
        "array_sort_indices",
        [inp.to_arrow()],
        options=pc.ArraySortOptions(
            order="descending" if descending else "ascending",
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(  # Safe to ignore.
            "ignore",
            message="The given NumPy array is not writable",
        )
        perm = torch.from_numpy(out.to_numpy()).to(inp.device, torch.int64)

    return cast(StringTensor, inp[perm]), perm
