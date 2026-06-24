import math
from collections.abc import Sequence
from typing import Any, ClassVar, cast

import pyarrow as pa
import torch

from sdm.tensor import VarLenTensor


class StringTensor(VarLenTensor):
    # NOTE Assume that `data` stores valid UTF-8 bytes and do not validate it.
    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = (torch.uint8,)

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> "StringTensor":

        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks == 1:
                array = array.chunk(0)
            else:
                array = array.combine_chunks()

        if not isinstance(array, pa.Array):
            raise TypeError(
                f"Expected 'array' in '{cls.__name__}.from_arrow' to be a "
                f"'pyarrow.Array' or 'pyarrow.ChunkedArray' "
                f"(got '{type(array).__name__}')"
            )

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

    def to_arrow(self) -> pa.Array:
        if self.device.type != "cpu":
            raise TypeError(
                f"Can't convert {self.device} device type tensor to arrow. "
                f"Use 'Tensor.cpu()' to copy the tensor to host memory first."
            )
        if self.requires_grad:
            raise RuntimeError(
                "Can't call 'to_arrow()' on Tensor that requires grad. "
                "Use 'Tensor.detach().to_arrow()' instead."
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

    @classmethod
    def from_list(
        cls,
        values: str | Sequence[Any],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> "StringTensor":

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

    def item(self) -> str:  # type: ignore
        return cast(str, super().item())

    def __str__(self) -> str:
        return self.item() if self.numel() == 1 else self.__repr__()

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}(..."
        out += f", size={tuple(self.size())}"
        if self.device.type != "cpu":
            out += f", device={self.device}"
        if self._data.grad_fn is not None:
            out += f", grad_fn=<{type(self._data.grad_fn).__name__}>"
        elif self.requires_grad:
            out += ", requires_grad=True>"
        out += ")"
        return out
