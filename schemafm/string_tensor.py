import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

aten = torch.ops.aten

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def implements(torch_function: Callable[..., Any]) -> Callable[..., Any]:

    def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


class StringTensor(Tensor):
    _data: Tensor
    _offset: Tensor

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    def __init__(
        cls,
        data: Tensor,
        offset: Tensor,
        size: Sequence[int],
        *,
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> None:
        pass

    @staticmethod
    def __new__(
        cls,
        data: Tensor,
        offset: Tensor,
        size: Sequence[int],
        *,
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> "StringTensor":

        if stride is None:
            stride, value = [], 1
            for dim_size in reversed(size):
                stride.append(value)
                value *= dim_size
            stride = stride[::-1]

        if data.dtype != torch.uint8:
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to have dtype "
                f"'torch.uint8' (got '{data.dtype}')"
            )
        if data.dim() != 1:
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to be one-dimensional "
                f"(got {data.dim()}D tensor)"
            )
        if offset.dtype != torch.long:  # TODO Relax 8-byte offset restriction.
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to have dtype "
                f"'torch.int64' (got '{offset.dtype}')"
            )
        if offset.dim() != 1:
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to be one-dimensional "
                f"(got {offset.dim()}D tensor)"
            )
        if data.device != offset.device:
            raise ValueError(
                f"Expected 'data' and 'offset' in '{cls.__name__}' to be on "
                f"the same device (got '{data.device}' and '{offset.device}')"
            )
        if len(size) != len(stride):
            raise ValueError(
                f"Expected 'size' and 'stride' in '{cls.__name__}' to have "
                f"the same length (got {len(size)} and {len(stride)})"
            )
        if storage_offset < 0:
            raise ValueError(
                f"Expected 'storage_offset' in '{cls.__name__}' to be "
                f"non-negative"
            )
        if math.prod(size) > 0:
            max_index = storage_offset + sum(
                (dim_size - 1) * dim_stride
                for dim_size, dim_stride in zip(size, stride, strict=True)
            )
            if max_index >= offset.numel() - 1:
                raise ValueError(
                    f"'offset' in '{cls.__name__}' is out of bounds (got "
                    f"{offset.numel() - 1} elements, but expected at least "
                    f"{max_index + 1} elements)"
                )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=tuple(size),
            strides=tuple(stride),
            storage_offset=storage_offset,
            dtype=torch.uint8,
            device=data.device,
            layout=torch.strided,
            requires_grad=False,
        )

        out._data = data
        out._offset = offset

        return out

    @classmethod
    def from_list(
        cls,
        data: str | Sequence[Any],
        *,
        device: torch.device | str | None = None,
    ) -> "StringTensor":
        values = bytearray()
        offsets = [0]

        def flatten(data: Any) -> tuple[int, ...]:
            if isinstance(data, str):
                values.extend(data.encode("utf-8"))
                offsets.append(len(values))
                return ()
            if not isinstance(data, Sequence):
                raise TypeError(f"'{cls.__name__}' data must contain strings")
            if len(data) == 0:
                return (0,)

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

        size = flatten(data)

        return cls(
            data=torch.tensor(values, dtype=torch.uint8, device=device),
            offset=torch.tensor(offsets, dtype=torch.long, device=device),
            size=size,
        )

    @property
    def bytes(self) -> Tensor:
        return self._data

    @property
    def offset(self) -> Tensor:
        return self._offset

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ["_data", "_offset"]
        ctx = (self.storage_offset(),)
        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> "StringTensor":
        (storage_offset,) = ctx
        return StringTensor(
            data=inner_tensors["_data"],
            offset=inner_tensors["_offset"],
            size=outer_size,
            stride=outer_stride,
            storage_offset=storage_offset,
        )

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for '{cls.__name__}'"
        )
