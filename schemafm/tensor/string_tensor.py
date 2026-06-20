import math
from collections.abc import Sequence
from typing import Any, ClassVar, cast

import pyarrow as pa
import torch

from schemafm.tensor import VarLenTensor


class StringTensor(VarLenTensor):
    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = (torch.uint8,)

    @classmethod
    def from_arrow(
        cls,
        data: pa.Array | pa.ChunkedArray,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> "StringTensor":
        if isinstance(data, pa.ChunkedArray):
            if data.num_chunks == 1:
                data = data.chunk(0)
            else:
                data = data.combine_chunks()

        if not isinstance(data, pa.Array):
            raise TypeError(
                f"Expected 'data' in '{cls.__name__}.from_arrow' to be a "
                f"'pyarrow.Array' or 'pyarrow.ChunkedArray' "
                f"(got '{type(data).__name__}')"
            )

        is_string = pa.types.is_string(data.type)
        is_large_string = pa.types.is_large_string(data.type)
        if not is_string and not is_large_string:
            raise TypeError(
                f"Expected 'data' in '{cls.__name__}.from_arrow' to have "
                f"'string' or 'large_string' type (got '{data.type}')"
            )

        if size is None:
            size = (len(data),)
        elif math.prod(size) != len(data):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_arrow' to contain "
                f"{len(data)} elements (got {math.prod(size)})"
            )

        buffers = data.buffers()

        return cls(
            data=torch.frombuffer(buffers[2], dtype=torch.uint8).to(device)
            if buffers[2].size > 0
            else torch.empty(0, dtype=torch.uint8, device=device),
            offset=torch.frombuffer(
                buffer=buffers[1],
                dtype=torch.int32 if is_string else torch.int64,
            ).to(device),
            size=size,
            storage_offset=data.offset,
        )

    @classmethod
    def from_strings(
        cls,
        data: str | Sequence[Any],
        *,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> "StringTensor":

        if offset_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Expected 'offset_dtype' in '{cls.__name__}.from_strings' "
                f"to be 'torch.int32' or 'torch.int64' "
                f"(got '{offset_dtype}')"
            )

        def flatten(data: Any) -> tuple[int, ...]:
            if isinstance(data, str):
                return ()
            if not isinstance(data, Sequence):
                raise TypeError(f"'{cls.__name__}' data must contain strings")
            if len(data) == 0:
                return (0,)

            if not isinstance(data[0], Sequence) or isinstance(data[0], str):
                values.extend(data)
                return (len(data),)

            child_size: tuple[int, ...] | None = None
            for item in data:
                item_size = flatten(item)
                if child_size is None:
                    child_size = item_size
                elif item_size != child_size:
                    raise ValueError(
                        f"'{cls.__name__}' data must be rectangular"
                    )

            assert child_size is not None
            return (len(data), *child_size)

        if isinstance(data, str):
            values: list[str] = [data]
            size: tuple[int, ...] = ()
        else:
            values = []
            size = flatten(data)

        pa_type = pa.large_string()
        if offset_dtype == torch.int32:
            pa_type = pa.string()

        return cls.from_arrow(
            data=pa.array(values, type=pa_type),
            device=device,
            size=size,
        )

    @classmethod
    def from_pandas(
        cls,
        data: Any,
        *,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> "StringTensor":
        import pandas as pd

        if not isinstance(data, pd.Series):
            raise TypeError(
                f"Expected 'data' in '{cls.__name__}.from_pandas' to be a "
                f"'pandas.Series' (got '{type(data).__name__}')"
            )
        if offset_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Expected 'offset_dtype' in '{cls.__name__}.from_pandas' "
                f"to be 'torch.int32' or 'torch.int64' "
                f"(got '{offset_dtype}')"
            )

        pa_type = pa.large_string()
        if offset_dtype == torch.int32:
            pa_type = pa.string()

        return cls.from_arrow(
            data=data.astype(pd.ArrowDtype(pa_type)).array.__arrow_array__(),
            device=device,
        )

    def to_arrow(self) -> pa.Array:
        if self.device.type != "cpu":
            raise TypeError(
                f"can't convert {self.device} device type tensor to arrow. "
                f"Use Tensor.cpu() to copy the tensor to host memory first."
            )

        data, offset = cast(StringTensor, self.contiguous()).data_offset

        return pa.Array.from_buffers(
            pa.string() if offset.dtype == torch.int32 else pa.large_string(),
            length=self.numel(),
            buffers=[
                None,
                pa.py_buffer(offset.numpy()),
                pa.py_buffer(data.numpy()),
            ],
        )

    def to_pandas(self) -> Any:
        if self.dim() != 1:
            raise ValueError(
                f"Expected '{self.__class__.__name__}' to be "
                f"one-dimensional for 'to_pandas' (got {self.dim()}D tensor)"
            )

        return self.to_arrow().to_pandas()

    def item(self) -> str:  # type: ignore
        if self.numel() != 1:
            raise RuntimeError(
                f"a Tensor with {self.numel()} elements cannot be converted "
                f"to a string"
            )

        start = self._offset[int(self.storage_offset())]
        end = self._offset[int(self.storage_offset()) + 1]
        return bytes(self._data[start:end].tolist()).decode("utf-8")

    def tolist(self) -> str | list[Any]:  # type: ignore
        def reshape(seq: list[str], size: tuple[int, ...]) -> str | list[Any]:
            if len(size) == 0:
                return seq[0]
            if len(size) == 1:
                return seq

            step = math.prod(size[1:])
            return [
                reshape(seq[i : i + step], size[1:])
                for i in range(0, len(seq), step)
            ]

        return reshape(self.to_arrow().to_pylist(), tuple(self.size()))

    def __str__(self) -> str:
        return self.item() if self.numel() == 1 else self.__repr__()
