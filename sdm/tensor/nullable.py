from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, cast

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import Self, override

from sdm.tensor.io import (
    ARROW_TORCH_DTYPES,
    TORCH_ARROW_DTYPES,
    arrow_as_tensor,
    to_cudf,
)
from sdm.tensor.io.arrow import _combine_arrow_chunks

if TYPE_CHECKING:
    import cudf


class NullableIntTensor(Tensor):
    r"""A :class:`torch.Tensor` for nullable integer values.

    Args:
        data: Integer tensor containing the values.
        valid: Boolean mask indicating valid, non-null values.
    """

    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...]] = (
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

    _data: Tensor
    _valid: Tensor | None

    # Constructors ############################################################

    def __init__(
        self,
        data: Tensor,
        valid: Tensor | None = None,
    ) -> None:
        pass

    def __new__(
        cls,
        data: Tensor,
        valid: Tensor | None = None,
    ) -> Self:
        r"""Create a tensor wrapper."""
        cls._validate_data(data)
        cls._validate_valid(data, valid)

        out = Tensor._make_wrapper_subclass(
            cls,
            size=data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            device=data.device,
            requires_grad=False,
        )

        out._data = data
        out._valid = valid

        return out

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        dtype: torch.dtype | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from an integer :class:`pyarrow.Array`.

        Args:
            array: The integer :class:`pyarrow.Array` or
                :class:`pyarrow.ChunkedArray`.
            dtype: The dtype.
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

        if not pa.types.is_integer(array.type):
            raise TypeError(
                f"Expected 'array' in '{cls.__name__}.from_arrow' to have "
                f"integer type (got '{array.type}')"
            )

        arrow_dtype = ARROW_TORCH_DTYPES.get(array.type)
        if arrow_dtype is None or arrow_dtype not in cls.ALLOWED_DTYPES:
            raise TypeError(f"Unsupported value type '{array.type}'")
        if dtype is not None:
            cls._validate_dtype(dtype, name="dtype")

        buffer = array.buffers()[1]
        if buffer is not None and buffer.size > 0:
            data = torch.frombuffer(buffer, dtype=arrow_dtype)
            data = data[array.offset : array.offset + len(array)]
            if array.offset != 0:
                data = data.contiguous()
            data = data.to(device=device, dtype=dtype)
        else:
            data = torch.empty(
                len(array),
                dtype=arrow_dtype if dtype is None else dtype,
                device=device,
            )

        valid: Tensor | None = None
        if array.null_count > 0:
            valid = arrow_as_tensor(
                array.is_valid(),
                dtype=torch.bool,
                device=device,
            )

        return cls(
            data=data.view(size),
            valid=valid.view(size) if valid is not None else None,
        )

    def to_arrow(self) -> pa.Array:
        r"""Convert this tensor to a flat :class:`pyarrow.Array`."""
        tensor = self.contiguous().cpu()
        data = tensor._data.view(-1)
        valid = tensor._valid.view(-1) if tensor._valid is not None else None

        arrow_type = TORCH_ARROW_DTYPES.get(data.dtype)
        if arrow_type is None:
            raise TypeError(f"Unsupported data type '{data.dtype}'")

        return pa.Array.from_buffers(
            type=arrow_type,
            length=data.numel(),
            buffers=[
                pa.array(valid.numpy(), type=pa.bool_()).buffers()[1]
                if valid is not None
                else None,
                pa.py_buffer(data.numpy()),
            ],
            null_count=-1,
        )

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series | cudf.Index,
        *,
        dtype: torch.dtype | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from an integer :class:`cudf.Series`.

        Args:
            ser: The integer :class:`cudf.Series` or :class:`cudf.Index`.
            dtype: The dtype.
            size: The shape of the tensor.
            device: The device.
        """
        from cudf.api.types import is_integer_dtype

        if size is None:
            size = (len(ser),)
        elif math.prod(size) != len(ser):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_cudf' to contain "
                f"{len(ser)} elements (got {math.prod(size)})"
            )

        if not is_integer_dtype(ser.dtype):
            raise TypeError(
                f"Expected 'ser' in '{cls.__name__}.from_cudf' to have "
                f"integer type (got '{ser.dtype}')"
            )
        if dtype is not None:
            cls._validate_dtype(dtype, name="dtype")

        valid: Tensor | None = None
        data_ser = ser
        if ser.hasnans:
            data_ser = ser.fillna(0)
            valid = torch.from_dlpack(ser.notnull().to_cupy()).to(device)

        data = torch.from_dlpack(data_ser.to_dlpack()).to(
            device=device,
            dtype=dtype,
        )

        return cls(
            data=data.view(size),
            valid=valid.view(size) if valid is not None else None,
        )

    def to_cudf(self) -> cudf.Series:
        r"""Convert this CUDA tensor to a flat :class:`cudf.Series`."""
        return to_cudf(self._data, self._valid)

    @classmethod
    def from_list(
        cls,
        values: int | None | Sequence[Any],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a rectangular Python list of integers.

        Args:
            values: The rectangular Python list of integers.
            dtype: The dtype.
            device: The device.
        """
        if dtype is not None:
            cls._validate_dtype(dtype, name="dtype")

        data: list[Any] = []
        valid: list[bool] = []

        def is_sequence(value: Any) -> bool:
            return isinstance(value, Sequence) and not isinstance(
                value, str | bytes | bytearray
            )

        def flatten(value: Any) -> tuple[int, ...]:
            if value is None:
                data.append(0)
                valid.append(False)
                return ()

            if not is_sequence(value):
                data.append(value)
                valid.append(True)
                return ()

            if len(value) == 0:
                return (0,)

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
        data_dtype = torch.int64 if dtype is None and len(data) == 0 else dtype
        tensor = torch.tensor(data, dtype=data_dtype, device=device)
        valid_tensor = None
        if False in valid:
            valid_tensor = torch.tensor(
                valid,
                dtype=torch.bool,
                device=device,
            )

        return cls(
            data=tensor.view(size),
            valid=valid_tensor.view(size)
            if valid_tensor is not None
            else None,
        )

    # Properties ##############################################################

    @property
    def values(self) -> Tensor:
        r"""Return the integer values tensor."""
        return self._data

    @property
    def valid(self) -> Tensor | None:
        r"""Return the validity mask, or ``None`` when all values are valid."""
        return self._valid

    @property
    def is_nullable(self) -> bool:
        r"""Whether this tensor has a validity mask."""
        return self._valid is not None

    # PyTorch/Python builtins #################################################

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        return (self.__class__, (self._data, self._valid))

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError(
            f"'{func}' is not supported for {cls.__name__!r}"
        )

    @override
    def is_shared(self) -> bool:
        is_shared = self._data.is_shared()
        return is_shared and (self._valid is None or self._valid.is_shared())

    @override
    def share_memory_(self) -> Self:
        self._data.share_memory_()
        if self._valid is not None:
            self._valid.share_memory_()
        return self

    @override
    def is_pinned(self) -> bool:
        is_pinned = self._data.is_pinned()
        return is_pinned and (self._valid is None or self._valid.is_pinned())

    @override
    def pin_memory(self, device: Any = None) -> Self:
        if device is None:
            return self.__class__(
                data=self._data.pin_memory(),
                valid=self._valid.pin_memory()
                if self._valid is not None
                else None,
            )
        return self.__class__(
            data=self._data.pin_memory(device),
            valid=self._valid.pin_memory(device)
            if self._valid is not None
            else None,
        )

    @override
    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        is_contiguous = self._data.is_contiguous(
            memory_format=memory_format,
        )
        return is_contiguous and (
            self._valid is None
            or self._valid.is_contiguous(memory_format=memory_format)
        )

    @override
    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Self:
        if self.is_contiguous(memory_format=memory_format):
            return self
        return self.__class__(
            data=self._data.contiguous(memory_format=memory_format),
            valid=self._valid.contiguous(memory_format=memory_format)
            if self._valid is not None
            else None,
        )

    @override
    def clone(
        self,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self:
        return self.__class__(
            data=self._data.clone(memory_format=memory_format),
            valid=self._valid.clone(memory_format=memory_format)
            if self._valid is not None
            else None,
        )

    @override
    def detach(self) -> Self:
        return self.__class__(
            data=self._data.detach(),
            valid=self._valid.detach() if self._valid is not None else None,
        )

    @override
    def detach_(self) -> Self:
        raise RuntimeError(
            f"Can't detach a '{self.__class__.__name__}' in-place. Use "
            "'detach() instead."
        )

    @override
    def cpu(self) -> Self:
        return self.__class__(
            data=self._data.cpu(),
            valid=self._valid.cpu() if self._valid is not None else None,
        )

    @override
    def tolist(self) -> Any:
        values = self._data.cpu().tolist()
        valid = self._valid.cpu().tolist() if self._valid is not None else None
        if valid is None:
            return values

        return self._apply_valid(values, valid)

    @override
    def item(self) -> int | None:  # type: ignore
        if self._data.numel() != 1:
            raise RuntimeError(
                f"{self.__class__.__name__!r} with {self._data.numel()} "
                "elements cannot be converted to a single item"
            )
        return cast(int | None, self.tolist())

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}(..."
        out += f", size={tuple(self.size())}"
        out += f", dtype={self.dtype}"
        if self._valid is not None:
            out += f", null_count={int((~self._valid).sum())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        out += ")"
        return out

    # Helpers ################################################################

    @classmethod
    def _validate_data(cls, data: Tensor) -> None:
        cls._validate_dtype(data.dtype, name="data")
        if not data.is_contiguous():
            raise ValueError(
                f"Expected 'data' in {cls.__name__!r} to be contiguous"
            )

    @classmethod
    def _validate_valid(cls, data: Tensor, valid: Tensor | None) -> None:
        if valid is None:
            return
        if valid.dtype != torch.bool:
            raise ValueError(
                f"Expected 'valid' in {cls.__name__!r} to have dtype "
                f"'torch.bool' (got '{valid.dtype}')"
            )
        if valid.size() != data.size():
            raise ValueError(
                f"Expected 'valid' in {cls.__name__!r} to have size "
                f"{tuple(data.size())} (got {tuple(valid.size())})"
            )
        if not valid.is_contiguous():
            raise ValueError(
                f"Expected 'valid' in {cls.__name__!r} to be contiguous"
            )
        if valid.device != data.device:
            raise ValueError(
                f"Expected 'data' and 'valid' in {cls.__name__!r} to be on "
                f"the same device (got '{data.device}' and '{valid.device}')"
            )

    @classmethod
    def _validate_dtype(cls, dtype: torch.dtype, *, name: str) -> None:
        if dtype not in cls.ALLOWED_DTYPES:
            raise ValueError(
                f"Expected '{name}' in {cls.__name__!r} to have integer "
                f"dtype (got '{dtype}')"
            )

    @classmethod
    def _apply_valid(cls, values: Any, valid: Any) -> Any:
        if isinstance(valid, bool):
            return values if valid else None
        return [
            cls._apply_valid(value, is_valid)
            for value, is_valid in zip(values, valid)
        ]
