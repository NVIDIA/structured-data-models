from collections.abc import Callable, Sequence
from typing import Any, ClassVar, SupportsIndex, TypeVar

import torch
from torch import Tensor
from torch.utils import _pytree as pytree

aten = torch.ops.aten

SelfCategoricalTensor = TypeVar(
    "SelfCategoricalTensor",
    bound="CategoricalTensor",
)


class CategoricalTensor(Tensor):
    # Negative data values represent missing values. Valid data values are
    # direct indices into the corresponding category vector.
    ALLOWED_DTYPES = (torch.int32, torch.int64)
    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _data: Tensor
    _categories: tuple[Tensor, ...]

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        cls,
        data: Tensor,
        categories: Sequence[Tensor],
    ) -> None:
        pass

    def __new__(
        cls: type[SelfCategoricalTensor],
        data: Tensor,
        categories: Sequence[Tensor],
    ) -> SelfCategoricalTensor:

        if data.dtype not in cls.ALLOWED_DTYPES:
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to have dtype "
                f"in '{cls.ALLOWED_DTYPES}' (got '{data.dtype}')"
            )
        if data.dim() == 0:
            raise ValueError(
                f"Expected '{cls.__name__}' to have at least one dimension"
            )
        if data.size(-1) != len(categories):
            raise ValueError(
                f"Expected the last dimension in '{cls.__name__}' to match "
                f"the number of category vectors (got {data.size(-1)} and "
                f"{len(categories)})"
            )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            device=data.device,
            layout=torch.strided,
            requires_grad=False,
        )

        out._data = data
        out._categories = tuple(categories)

        return out

    # Properties ##############################################################

    def as_tensor(self) -> Tensor:
        return self._data

    @property
    def categories(self) -> tuple[Tensor, ...]:
        return self._categories

    # Decorators ##############################################################

    @classmethod
    def implements(
        cls,
        torch_function: Callable[..., Any],
    ) -> Callable[..., Any]:
        if "HANDLED_FUNCTIONS" not in cls.__dict__:
            cls.HANDLED_FUNCTIONS = cls.HANDLED_FUNCTIONS.copy()

        def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
            cls.HANDLED_FUNCTIONS[torch_function] = my_function
            return my_function

        return decorator

    # PyTorch/Python builtins #################################################

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (self.__class__, self._data, self._categories)
        return (_deserialize, args)

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

        # Operate on vanilla tensors for all non-handled functions:
        args = pytree.tree_map_only(CategoricalTensor, lambda x: x._data, args)
        kwargs = pytree.tree_map_only(
            CategoricalTensor, lambda x: x._data, kwargs
        )
        return func(*args, **(kwargs or {}))

    def is_shared(self) -> bool:
        return self._data.is_shared()

    def share_memory_(self) -> "CategoricalTensor":
        self._data.share_memory_()
        return self


@CategoricalTensor.implements(aten.isnan.default)
def _isnan(input: CategoricalTensor) -> Tensor:
    return input._data < 0


@CategoricalTensor.implements(aten._to_copy.default)
def _to_copy(
    input: CategoricalTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:

    data = aten._to_copy.default(
        input._data,
        device=device,
        dtype=dtype,
        layout=layout,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )
    if data.dtype not in input.ALLOWED_DTYPES or data.layout != torch.strided:
        return data

    categories = tuple(
        category.to(device=device, non_blocking=non_blocking, copy=True)
        for category in input._categories
    )
    return input.__class__(data, categories)


@CategoricalTensor.implements(aten.clone.default)
def _clone(
    input: CategoricalTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> CategoricalTensor:
    out = _to_copy(input, memory_format=memory_format)
    assert isinstance(out, CategoricalTensor)
    return out


@CategoricalTensor.implements(aten.contiguous.default)
def _contiguous(
    input: CategoricalTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> CategoricalTensor:
    data = input._data.contiguous(memory_format=memory_format)
    return input.__class__(data, input._categories)


@CategoricalTensor.implements(aten._pin_memory.default)
def _pin_memory(input: CategoricalTensor) -> CategoricalTensor:
    return input.__class__(
        input._data.pin_memory(),
        input._categories,
    )


@CategoricalTensor.implements(aten.view.default)
def _view(
    input: CategoricalTensor,
    size: Sequence[int],
) -> CategoricalTensor | Tensor:
    data = input._data.view(tuple(size))
    return _wrap_if_category_dim_preserved(input, data)


@CategoricalTensor.implements(aten._unsafe_view.default)
def _unsafe_view(
    input: CategoricalTensor,
    size: Sequence[int],
) -> CategoricalTensor | Tensor:
    data = aten._unsafe_view.default(input._data, size)
    return _wrap_if_category_dim_preserved(input, data)


@CategoricalTensor.implements(aten.squeeze.default)
def _squeeze(input: CategoricalTensor) -> CategoricalTensor | Tensor:
    data = input._data.squeeze()
    return _wrap_if_category_dim_preserved(input, data)


@CategoricalTensor.implements(aten.squeeze.dim)
def _squeeze_dim(
    input: CategoricalTensor,
    dim: int,
) -> CategoricalTensor | Tensor:
    data = input._data.squeeze(dim)
    return _wrap_if_category_dim_preserved(input, data)


@CategoricalTensor.implements(aten.squeeze.dims)
def _squeeze_dims(
    input: CategoricalTensor,
    dim: Sequence[int],
) -> CategoricalTensor | Tensor:
    data = aten.squeeze.dims(input._data, dim)
    return _wrap_if_category_dim_preserved(input, data)


@CategoricalTensor.implements(aten.unsqueeze.default)
def _unsqueeze(
    input: CategoricalTensor,
    dim: int,
) -> CategoricalTensor | Tensor:
    data = input._data.unsqueeze(dim)
    dim = _normalize_dim(dim, input.dim() + 1)
    if dim < input.dim():
        return input.__class__(data, input._categories)
    return data


@CategoricalTensor.implements(aten.expand.default)
def _expand(
    input: CategoricalTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> CategoricalTensor | Tensor:
    data = aten.expand.default(input._data, size, implicit=implicit)
    return _wrap_if_category_dim_preserved(input, data)


@CategoricalTensor.implements(aten.transpose.int)
def _transpose(
    input: CategoricalTensor,
    dim0: int,
    dim1: int,
) -> CategoricalTensor | Tensor:
    data = input._data.transpose(dim0, dim1)
    dim0 = _normalize_dim(dim0, input.dim())
    dim1 = _normalize_dim(dim1, input.dim())
    category_dim = input.dim() - 1
    if dim0 != dim1 and category_dim in (dim0, dim1):
        return data
    return input.__class__(data, input._categories)


@CategoricalTensor.implements(aten.permute.default)
def _permute(
    input: CategoricalTensor,
    dims: Sequence[int],
) -> CategoricalTensor | Tensor:
    data = input._data.permute(tuple(dims))
    dims = tuple(_normalize_dim(dim, input.dim()) for dim in dims)
    if dims[-1] == input.dim() - 1:
        return input.__class__(data, input._categories)
    return data


# Helpers #####################################################################


def _deserialize(
    cls: type[SelfCategoricalTensor],
    data: Tensor,
    categories: tuple[Tensor, ...],
) -> SelfCategoricalTensor:
    return cls(data, categories)


def _wrap_if_category_dim_preserved(
    input: CategoricalTensor,
    data: Tensor,
) -> CategoricalTensor | Tensor:
    if (
        data.layout == torch.strided
        and data.dtype in input.ALLOWED_DTYPES
        and data.dim() > 0
        and data.size(-1) == len(input._categories)
    ):
        return input.__class__(data, input._categories)
    return data


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{ndim}, {ndim - 1}], but got {dim})"
        )
    return dim
