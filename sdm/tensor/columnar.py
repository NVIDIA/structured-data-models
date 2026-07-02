from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, ClassVar, SupportsIndex, TypeVar, cast

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import override

from sdm.tensor import CategoricalTensor, StringTensor

aten = torch.ops.aten

SelfColumnarTensor = TypeVar(
    "SelfColumnarTensor",
    bound="ColumnarTensor",
)


class ColumnarTensor(Tensor):
    r"""A :class:`torch.Tensor` for column-wise heterogeneous data.

    A :class:`ColumnarTensor` exposes a tensor-centric interface for columnar
    data with shape ``[..., C]``.
    Each of the ``C`` columns is stored independently and may use a different
    tensor subclass or dtype.

    Args:
        columns: Per-column values.
        size: The shape of the tensor ``[...]``.
        device: The device.
    """

    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _columns: tuple[Tensor, ...]

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        self,
        columns: Sequence[Tensor],
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        pass

    def __new__(
        cls: type[SelfColumnarTensor],
        columns: Sequence[Tensor],
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> SelfColumnarTensor:
        r"""Create a tensor wrapper."""
        columns = tuple(columns)
        device = torch.device(device) if device is not None else None

        if size is None:
            if len(columns) == 0:
                raise ValueError(
                    "Expected 'size' to be given for zero columnar data"
                )
            if isinstance(columns[0], CategoricalTensor):
                size = tuple(columns[0].size()[:-1])
            else:
                size = tuple(columns[0].size())
        else:
            size = tuple(size)

        if len(size) < 1:
            raise ValueError(
                f"Expected 'size' to have at least one dimension "
                f"(got {len(size)})"
            )

        for i, column in enumerate(columns):
            if isinstance(column, CategoricalTensor):
                if column.size(-1) != 1:
                    raise ValueError(
                        f"Expected categorical tensor {i} in '{cls.__name__}' "
                        f"to only hold a single column"
                    )
                column_size = column.size()[:-1]
            else:
                column_size = column.size()
            if column_size != size:
                raise ValueError(
                    f"Expected value {i} in '{cls.__name__}' to have size "
                    f"{size} (got {tuple(column_size)})"
                )

            device = column.device if device is None else device
            if column.device != device:
                raise ValueError(
                    f"Expected value {i} in '{cls.__name__}' to be on "
                    f"device '{device}' (got '{column.device}')"
                )

        if device is None:
            device = torch.get_default_device()

        out = Tensor._make_wrapper_subclass(
            cls,
            size=(*size, len(columns)),
            dtype=torch.uint8,  # NOTE Do not use.
            device=device,
            requires_grad=False,
        )

        out._columns = columns

        return out

    @classmethod
    def from_arrow(
        cls: type[SelfColumnarTensor],
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | str | None = None,
    ) -> SelfColumnarTensor:
        r"""Create tensor from a ``pyarrow`` array.

        Args:
            array: The ``pyarrow`` array.
            device: The device.
        """
        device = torch.device("cpu" if device is None else device)

        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks == 1:
                array = array.chunk(0)
            else:
                array = array.combine_chunks()

        if array.null_count > 0:
            raise ValueError(f"'{cls.__name__}' cannot represent null values")

        is_string = pa.types.is_string(array.type)
        is_large_string = pa.types.is_large_string(array.type)
        if not is_string and not is_large_string:
            column = StringTensor.from_arrow(array, device=device)
        else:
            values = array.to_numpy(
                zero_copy_only=False,
                writable=device.type == "cpu",
            )
            column = torch.as_tensor(values, device=device)

        return cls(columns=(column,), device=device)

    # Decorators ##############################################################

    @classmethod
    def implements(
        cls,
        torch_function: Callable[..., Any],
    ) -> Callable[..., Any]:
        r"""Register a ``__torch_dispatch__`` implementation."""
        if "HANDLED_FUNCTIONS" not in cls.__dict__:
            cls.HANDLED_FUNCTIONS = cls.HANDLED_FUNCTIONS.copy()

        def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
            cls.HANDLED_FUNCTIONS[torch_function] = my_function
            return my_function

        return decorator

    # PyTorch/Python builtins #################################################

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (self._columns, tuple(self.size())[:-1], self.device)
        return (self.__class__, args)

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if (handler := cls.HANDLED_FUNCTIONS.get(func)) is not None:
            return handler(*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for '{cls.__name__}'"
        )

    @override
    def is_shared(self) -> bool:
        return all(column.is_shared() for column in self._columns)

    @override
    def share_memory_(self) -> "ColumnarTensor":
        for column in self._columns:
            column.share_memory_()
        return self

    @override
    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            column.is_contiguous(memory_format=memory_format)
            for column in self._columns
        )

    @override
    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> "ColumnarTensor":
        if self.is_contiguous(memory_format=memory_format):
            return self
        return _contiguous(self, memory_format=memory_format)

    @override
    def tolist(self) -> Any:
        def columns_to_rows(
            columns: Sequence[Any],
            size: tuple[int, ...],
        ) -> Any:
            if len(size) == 0:
                return list(columns)

            return [
                columns_to_rows(
                    columns=[column[i] for column in columns],
                    size=size[1:],
                )
                for i in range(size[0])
            ]

        columns = [column.tolist() for column in self._columns]
        return columns_to_rows(columns, size=tuple(self.size()[:-1]))

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        out = f"{self.__class__.__name__}("
        out += f"size={tuple(self.size())}"
        if self.device.type != "cpu":
            out += f", device={self.device}"
        out += ")"
        return out


@ColumnarTensor.implements(aten.alias.default)
def _alias(input: ColumnarTensor) -> ColumnarTensor:
    return input.__class__(
        columns=input._columns,
        size=input.size()[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten._to_copy.default)
def _to_copy(
    input: ColumnarTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:

    if dtype is not None:
        raise TypeError(
            f"Can't convert '{input.__class__.__name__}' to dtype '{dtype}'"
        )

    return input.__class__(
        columns=[
            aten._to_copy.default(
                column,
                device=device,
                dtype=None,
                layout=layout,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )
            for column in input._columns
        ],
        size=input.size()[:-1],
        device=device,
    )


@ColumnarTensor.implements(aten.clone.default)
def _clone(
    input: ColumnarTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> ColumnarTensor:
    out = _to_copy(input, memory_format=memory_format)
    assert isinstance(out, ColumnarTensor)
    return out


@ColumnarTensor.implements(aten.contiguous.default)
def _contiguous(
    input: ColumnarTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> ColumnarTensor:
    return input.__class__(
        columns=[
            column.contiguous(memory_format=memory_format)
            for column in input._columns
        ],
        size=input.size()[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.is_pinned.default)
def _is_pinned(input: ColumnarTensor) -> bool:
    return all(column.is_pinned() for column in input._columns)


@ColumnarTensor.implements(aten._pin_memory.default)
def _pin_memory(input: ColumnarTensor) -> ColumnarTensor:
    return input.__class__(
        columns=[column.pin_memory() for column in input._columns],
        size=input.size()[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> ColumnarTensor:
    if not all(isinstance(tensor, ColumnarTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be '{ColumnarTensor.__name__}' instances"
        )

    tensors = cast(Sequence[ColumnarTensor], tensors)
    dim %= tensors[0].dim()
    if dim == tensors[0].dim() - 1:
        return tensors[0].__class__(
            columns=tuple(chain.from_iterable(t._columns for t in tensors)),
            device=tensors[0].device,
        )

    return tensors[0].__class__(
        columns=[
            torch.cat([tensor._columns[i] for tensor in tensors], dim=dim)
            for i in range(tensors[0].size(-1))
        ],
        device=tensors[0].device,
    )
