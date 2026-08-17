from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm import VarLenTensor

aten = torch.ops.aten


def _is_alias(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _nullable_tensor() -> VarLenTensor:
    return VarLenTensor.from_list([[1.0], None, [2.0, 3.0]])


def test_to_dtype_layout_defaults_return_input() -> None:
    tensor = _nullable_tensor()

    assert aten.to.dtype_layout(tensor) is tensor


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda tensor: aten.to.dtype_layout(
                tensor,
                dtype=torch.float64,
                layout=torch.strided,
                device=torch.device("cpu"),
                pin_memory=False,
                non_blocking=False,
                copy=False,
                memory_format=torch.preserve_format,
            ),
            id="to.dtype_layout",
        ),
        pytest.param(
            lambda tensor: aten.to.dtype(
                tensor,
                torch.float64,
                False,
                False,
                torch.preserve_format,
            ),
            id="to.dtype",
        ),
        pytest.param(
            lambda tensor: aten.to.device(
                tensor,
                torch.device("cpu"),
                torch.float64,
                False,
                False,
                torch.preserve_format,
            ),
            id="to.device",
        ),
        pytest.param(
            lambda tensor: aten.to.other(
                tensor,
                torch.empty((), dtype=torch.float64),
                False,
                False,
                torch.preserve_format,
            ),
            id="to.other",
        ),
        pytest.param(
            lambda tensor: aten._to_copy.default(
                tensor,
                dtype=torch.float64,
                layout=torch.strided,
                device=torch.device("cpu"),
                pin_memory=False,
                non_blocking=False,
                memory_format=torch.preserve_format,
            ),
            id="_to_copy.default",
        ),
    ],
)
def test_to_overloads_convert_data_only(
    operation: Callable[[VarLenTensor], VarLenTensor],
) -> None:
    tensor = _nullable_tensor()
    out = operation(tensor)

    assert isinstance(out, VarLenTensor)
    assert out.tolist() == tensor.tolist()
    assert out.dtype == out._data.dtype == torch.float64
    assert out._offset.dtype == tensor._offset.dtype
    assert out._valid is not None
    assert out._valid.dtype == torch.bool


def test_clone_copies_all_inner_tensors() -> None:
    tensor = _nullable_tensor()
    out = aten.clone.default(
        tensor,
        memory_format=torch.preserve_format,
    )

    assert isinstance(out, VarLenTensor)
    assert out.tolist() == tensor.tolist()
    assert not _is_alias(out, tensor)
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()
    assert out._valid is not None
    assert tensor._valid is not None
    assert out._valid.data_ptr() != tensor._valid.data_ptr()


def test_contiguous_obeys_copy_elision() -> None:
    tensor = _nullable_tensor()
    assert (
        aten.contiguous.default(
            tensor,
            memory_format=torch.contiguous_format,
        )
        is tensor
    )

    view = VarLenTensor.from_list([[1.0], None, [2.0], [3.0]])
    view = view.view(2, 2).t()
    out = aten.contiguous.default(
        view,
        memory_format=torch.contiguous_format,
    )

    assert isinstance(out, VarLenTensor)
    assert out.is_contiguous()
    assert out.tolist() == view.tolist()
    assert not _is_alias(out, view)


def test_detach_preserves_type_and_aliases_storage() -> None:
    data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    tensor = VarLenTensor(
        data=data,
        offset=torch.tensor([0, 1, 3]),
        valid=None,
        size=(2,),
    )
    out = aten.detach.default(tensor)

    assert isinstance(out, VarLenTensor)
    assert not out.requires_grad
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert _is_alias(out, tensor)
