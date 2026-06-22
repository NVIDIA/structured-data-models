from collections.abc import Callable, Sequence
from itertools import accumulate, chain
from typing import Any, ClassVar, SupportsIndex, TypeVar, cast

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
def _view(input: CategoricalTensor, size: Sequence[int]) -> Tensor:
    return _maybe_wrap(input, input._data.view(size))


@CategoricalTensor.implements(aten._unsafe_view.default)
def _unsafe_view(input: CategoricalTensor, size: Sequence[int]) -> Tensor:
    return _maybe_wrap(input, aten._unsafe_view(input._data, size))


@CategoricalTensor.implements(aten.squeeze.default)
def _squeeze(input: CategoricalTensor) -> Tensor:
    return _maybe_wrap(input, input._data.squeeze())


@CategoricalTensor.implements(aten.squeeze.dim)
def _squeeze_dim(input: CategoricalTensor, dim: int) -> Tensor:
    return _maybe_wrap(input, input._data.squeeze(dim))


@CategoricalTensor.implements(aten.squeeze.dims)
def _squeeze_dims(input: CategoricalTensor, dim: Sequence[int]) -> Tensor:
    return _maybe_wrap(input, input._data.squeeze(tuple(dim)))


@CategoricalTensor.implements(aten.unsqueeze.default)
def _unsqueeze(input: CategoricalTensor, dim: int) -> Tensor:
    return _maybe_wrap(input, input._data.unsqueeze(dim))


@CategoricalTensor.implements(aten.expand.default)
def _expand(
    input: CategoricalTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> Tensor:
    data = aten.expand.default(input._data, size, implicit=implicit)
    return _maybe_wrap(input, data)


@CategoricalTensor.implements(aten.transpose.int)
def _transpose(input: CategoricalTensor, dim0: int, dim1: int) -> Tensor:
    data = input._data.transpose(dim0, dim1)
    dim0 %= input.dim()
    dim1 %= input.dim()
    if dim0 != dim1 and input.dim() - 1 in (dim0, dim1):
        return data
    return input.__class__(data, input.categories)


@CategoricalTensor.implements(aten.permute.default)
def _permute(input: CategoricalTensor, dims: Sequence[int]) -> Tensor:
    data = input._data.permute(tuple(dims))
    dims = tuple(dim % input.dim() for dim in dims)
    if dims[-1] != input.dim() - 1:
        return data
    return input.__class__(data, input.categories)


@CategoricalTensor.implements(aten.select.int)
def _select(input: CategoricalTensor, dim: int, index: int) -> Tensor:
    data = input._data.select(dim, index)
    dim %= input.dim()
    if dim == input.dim() - 1:
        return data
    return input.__class__(data, input.categories)


@CategoricalTensor.implements(aten.slice.Tensor)
def _slice(
    input: CategoricalTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> CategoricalTensor:
    data = aten.slice.Tensor(input._data, dim, start, end, step)
    dim %= input.dim()
    if dim != input.dim() - 1:
        return input.__class__(data, input.categories)
    return input.__class__(data, input.categories[slice(start, end, step)])


@CategoricalTensor.implements(aten.narrow.default)
def _narrow(
    input: CategoricalTensor,
    dim: int,
    start: int,
    length: int,
) -> CategoricalTensor:
    data = input._data.narrow(dim, start, length)
    dim %= input.dim()
    if dim != input.dim() - 1:
        return input.__class__(data, input.categories)
    if start < 0:
        start += input.size(dim)
    return input.__class__(data, input.categories[start : start + length])


@CategoricalTensor.implements(aten.unbind.int)
def _unbind(input: CategoricalTensor, dim: int = 0) -> tuple[Tensor, ...]:
    data_list = input._data.unbind(dim)
    dim %= input.dim()
    if dim == input.dim() - 1:
        return data_list
    return tuple(input.__class__(data, input.categories) for data in data_list)


@CategoricalTensor.implements(aten.split.Tensor)
def _split(
    input: CategoricalTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[CategoricalTensor, ...]:
    data_list = input._data.split(split_size, dim)
    dim %= input.dim()
    if dim != input.dim() - 1:
        return tuple(
            input.__class__(data, input.categories) for data in data_list
        )
    return tuple(
        input.__class__(data, input.categories[i : i + split_size])
        for data, i in zip(data_list, range(0, input.size(dim), split_size))
    )


@CategoricalTensor.implements(aten.split.sizes)
@CategoricalTensor.implements(aten.split.default)
@CategoricalTensor.implements(aten.split_with_sizes.default)
def _split_with_sizes(
    input: CategoricalTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[CategoricalTensor, ...]:
    data_list = input._data.split(tuple(split_sizes), dim)
    dim %= input.dim()
    if dim != input.dim() - 1:
        return tuple(
            input.__class__(data, input.categories) for data in data_list
        )

    offset = (0, *accumulate(split_sizes))
    return tuple(
        input.__class__(data, input.categories[start:end])
        for data, start, end in zip(data_list, offset[:-1], offset[1:])
    )


@CategoricalTensor.implements(aten.index_select.default)
def _index_select(
    input: CategoricalTensor,
    dim: int,
    index: Tensor,
) -> CategoricalTensor:
    data = input._data.index_select(dim, index)
    dim %= input.dim()
    if dim != input.dim() - 1:
        return input.__class__(data, input.categories)
    categories = tuple(input.categories[i] for i in index.tolist())
    return input.__class__(data, categories)


@CategoricalTensor.implements(aten.index.Tensor)
def _index(
    input: CategoricalTensor,
    indices: Sequence[Tensor | None],
) -> Tensor:
    data = aten.index.Tensor(input._data, indices)

    current_dim = 0
    has_other_index = False
    category_index: Tensor | None = None
    for index in indices:
        if index is None:
            current_dim += 1
            continue

        # Check whether we index the category dimension:
        num_indexed_dims = index.dim() if index.dtype == torch.bool else 1
        if current_dim <= input.dim() - 1 < current_dim + num_indexed_dims:
            if num_indexed_dims != 1:
                return data
            category_index = index
        else:
            has_other_index = True
        current_dim += num_indexed_dims

    if category_index is None:
        return _maybe_wrap(input, data)

    if has_other_index or category_index.dim() != 1:
        return data

    if category_index.dtype == torch.bool:
        category_index = category_index.nonzero().view(-1)

    categories = tuple(input.categories[i] for i in category_index.tolist())
    return input.__class__(data, categories)


@CategoricalTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    data = torch.cat([_as_tensor(tensor) for tensor in tensors], dim=dim)
    if not all(isinstance(tensor, CategoricalTensor) for tensor in tensors):
        return data

    tensors = cast(Sequence[CategoricalTensor], tensors)
    dim %= tensors[0].dim()
    if dim != tensors[0].dim() - 1:
        # NOTE We trust the user for category compatibility.
        return tensors[0].__class__(data, tensors[0].categories)

    categories = tuple(
        chain.from_iterable(tensor.categories for tensor in tensors)
    )
    return tensors[0].__class__(data, categories)


@CategoricalTensor.implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    data = torch.stack([_as_tensor(tensor) for tensor in tensors], dim=dim)
    if not all(isinstance(tensor, CategoricalTensor) for tensor in tensors):
        return data

    tensors = cast(Sequence[CategoricalTensor], tensors)
    dim %= tensors[0].dim() + 1
    if dim >= tensors[0].dim():
        return data

    return tensors[0].__class__(data, tensors[0].categories)


# Helpers #####################################################################


def _deserialize(
    cls: type[SelfCategoricalTensor],
    data: Tensor,
    categories: tuple[Tensor, ...],
) -> SelfCategoricalTensor:
    return cls(data, categories)


def _maybe_wrap(input: CategoricalTensor, data: Tensor) -> Tensor:
    if data.dim() > 0 and data.size(-1) == input.size(-1):
        return input.__class__(data, input.categories)
    return data


def _as_tensor(input: Tensor) -> Tensor:
    if isinstance(input, CategoricalTensor):
        return input._data
    return input


def _categories_equal(
    left: tuple[Tensor, ...],
    right: tuple[Tensor, ...],
) -> bool:
    if len(left) != len(right):
        return False

    try:
        return all(a.equal(b) for a, b in zip(left, right))
    except RuntimeError:
        return False
