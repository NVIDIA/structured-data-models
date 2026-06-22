from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypeVar

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


@CategoricalTensor.implements(aten.isnan.default)
def _isnan(input: CategoricalTensor) -> Tensor:
    return input._data < 0
