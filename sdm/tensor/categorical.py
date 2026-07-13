from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import accumulate, chain
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, cast

import pyarrow as pa
import torch
from torch import Tensor
from torch.utils import _pytree as pytree
from typing_extensions import Self, override

from sdm.tensor import StringTensor
from sdm.tensor._utils import _preserve_view_inference_mode
from sdm.tensor.io import to_arrow

aten = torch.ops.aten


if TYPE_CHECKING:
    import cudf


class CategoricalTensor(Tensor):
    r"""A :class:`torch.Tensor` for categorical column data.

    A :class:`CategoricalTensor` stores categorical indices in ``data`` and one
    category vector per column in ``categories``.
    Data values are direct indices into the corresponding category vector.
    Negative indices represent missing values.

    .. code-block:: python

        import torch
        from sdm import CategoricalTensor, StringTensor

        tensor = CategoricalTensor(
            data=torch.randint(0, 2, size=(10, 2)),
            categories=(
                StringTensor.from_list(["USA", "GERMANY"]),
                StringTensor.from_list(["enterprise", "startup"]),
            ),
        )

    Args:
        data: The categorical indices of shape ``[..., C]``.
        categories: A tuple of ``C`` category vectors.
    """

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
        self,
        data: Tensor,
        categories: Sequence[Tensor],
    ) -> None:
        pass

    def __new__(
        cls,
        data: Tensor,
        categories: Sequence[Tensor],
    ) -> Self:
        r"""Create a tensor wrapper."""
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
        for i, category in enumerate(categories):
            if category.dim() != 1:
                raise ValueError(
                    f"Expected category {i} in '{cls.__name__}' to be "
                    f"one-dimensional (got {category.dim()}D)"
                )

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
        out._categories = tuple(categories)

        return out

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a :class:`pyarrow.Array`.

        .. code-block:: python

            import pyarrow as pa
            from sdm import CategoricalTensor

            array = pa.array(["foo", None, "bar"])
            tensor = CategoricalTensor.from_arrow(array)

            print(tensor)
            >>> tensor([[ 0],
            >>>         [-1],
            >>>         [ 1]])
            print(tensor.categories[0].tolist())
            >>> ['foo', 'bar']

        Args:
            array: The :class:`pyarrow.Array` or
                :class:`pyarrow.ChunkedArray`.
            dtype: The dtype.
            device: The device.
        """
        device = torch.device("cpu" if device is None else device)

        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks == 1:
                array = array.chunk(0)
            else:
                array = array.combine_chunks()

        encoded = array.dictionary_encode()
        values = encoded.indices.fill_null(-1).to_numpy(
            zero_copy_only=False,
            writable=device.type == "cpu",
        )
        data = torch.as_tensor(
            values,
            dtype=dtype,
            device=device,
        ).unsqueeze(-1)

        dictionary = encoded.dictionary
        is_string = pa.types.is_string(dictionary.type)
        is_large_string = pa.types.is_large_string(dictionary.type)
        if is_string or is_large_string:
            category = StringTensor.from_arrow(dictionary, device=device)
        elif pa.types.is_null(dictionary.type):
            category = torch.empty(0, dtype=torch.int64, device=device)
        else:  # Use regular torch.Tensor for Tensor-compatible dictionaries:
            values = dictionary.to_numpy(
                zero_copy_only=False,
                writable=device.type == "cpu",
            )
            category = torch.as_tensor(
                values,
                device=device,
            )

        return cls(data=data, categories=(category,))

    def to_arrow(self, columns: Sequence[str] | None = None) -> pa.Table:
        r"""Convert this tensor to a flat :class:`pyarrow.Table`.

        Args:
            columns: Column names.
        """
        if columns is None:
            columns = tuple(str(i) for i in range(self.size(-1)))
        elif len(columns) != self.size(-1):
            raise ValueError(
                f"Expected 'columns' to contain {self.size(-1)} entries "
                f"(got {len(columns)})"
            )

        data_t = self._data.movedim(-1, 0).contiguous()

        arrays = []
        for data, category, na_mask in zip(
            data_t.cpu().unbind(0),
            self.categories,
            (data_t < 0).cpu().unbind(0),
        ):
            if na_mask.any().item():
                indices = pa.array(
                    data.clamp(min=0).view(-1).numpy(),
                    mask=na_mask.view(-1).numpy(),
                )
            else:
                indices = to_arrow(data.view(-1))

            arrays.append(
                pa.DictionaryArray.from_arrays(
                    indices=indices,
                    dictionary=to_arrow(category),
                )
            )

        return pa.Table.from_arrays(arrays, names=columns)

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series,
        *,
        dtype: torch.dtype = torch.int32,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a categorical :class:`cudf.Series`.

        Args:
            ser: The categorical :class:`cudf.Series`.
            dtype: The dtype.
            device: The device.
        """
        from cudf.api.types import is_string_dtype

        codes, categories = ser.factorize(
            sort=False,
            use_na_sentinel=True,
        )
        data = torch.from_dlpack(codes).unsqueeze(-1).to(device, dtype)

        if len(categories) == 0:
            category = torch.empty(0, dtype=torch.int64, device=data.device)
        elif is_string_dtype(categories.dtype):
            category = StringTensor.from_cudf(categories, device=device)
        else:
            category = torch.from_dlpack(categories.to_cupy()).to(device)

        return cls(data=data, categories=(category,))

    # Properties ##############################################################

    def as_tensor(self) -> Tensor:
        r"""Return the index tensor."""
        return self._data

    @property
    def categories(self) -> tuple[Tensor, ...]:
        r"""Return category vector for each categorical column."""
        return self._categories

    # Decorators ##############################################################

    @classmethod
    def implements(
        cls,
        torch_function: Callable[..., Any],
    ) -> Callable[..., Any]:
        r"""Register a ``__torch_dispatch__`` implementation.

        See PyTorch's
        :ref:`calling convention <torch-dispatch-calling-convention>`.
        """
        if "HANDLED_FUNCTIONS" not in cls.__dict__:
            cls.HANDLED_FUNCTIONS = cls.HANDLED_FUNCTIONS.copy()

        def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
            cls.HANDLED_FUNCTIONS[torch_function] = my_function
            return my_function

        return decorator

    # PyTorch/Python builtins #################################################

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (self._data, self._categories)
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
            with _preserve_view_inference_mode(func, args[0]):
                return handler(*args, **(kwargs or {}))

        # Operate on vanilla tensors for all non-handled functions:
        args = pytree.tree_map_only(CategoricalTensor, lambda x: x._data, args)
        kwargs = pytree.tree_map_only(
            CategoricalTensor, lambda x: x._data, kwargs
        )
        return func(*args, **(kwargs or {}))

    @override
    def is_shared(self) -> bool:
        return self._data.is_shared()

    @override
    def share_memory_(self) -> Self:
        self._data.share_memory_()
        return self

    @override
    def tolist(self) -> Any:
        def apply_na_mask(values: Any, na_mask: Any) -> Any:
            if isinstance(na_mask, bool):
                return None if na_mask else values

            return [
                apply_na_mask(value, isna)
                for value, isna in zip(values, na_mask)
            ]

        def decode_column(data: Tensor, category: Tensor) -> Any:
            na_mask = data < 0
            out = category[data.clamp(min=0)]
            return apply_na_mask(out.tolist(), na_mask.tolist())

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

        columns = [
            decode_column(self._data[..., i], category)
            for i, category in enumerate(self._categories)
        ]
        return columns_to_rows(columns, tuple(self.size()[:-1]))


@CategoricalTensor.implements(aten.isnan.default)
def _isnan(input: CategoricalTensor) -> Tensor:
    return input._data < 0


@CategoricalTensor.implements(aten.alias.default)
def _alias(input: CategoricalTensor) -> CategoricalTensor:
    return input.__class__(aten.alias.default(input._data), input._categories)


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
        aten._to_copy.default(
            category,
            device=device,
            dtype=None,
            layout=None,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=None,
        )
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
    return input.__class__(input._data.pin_memory(), input._categories)


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
        # NOTE We trust the user to ensure category compatibility.
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

    # NOTE We trust the user to ensure category compatibility.
    return tensors[0].__class__(data, tensors[0].categories)


# Helpers #####################################################################


def _maybe_wrap(input: CategoricalTensor, data: Tensor) -> Tensor:
    if data.dim() > 0 and data.size(-1) == input.size(-1):
        return input.__class__(data, input.categories)
    return data


def _as_tensor(input: Tensor) -> Tensor:
    if isinstance(input, CategoricalTensor):
        return input._data
    return input
