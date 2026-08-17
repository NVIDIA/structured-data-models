import torch
from torch import Tensor

from sdm import VarLenTensor

aten = torch.ops.aten


class EqualTensor(Tensor):
    @classmethod
    def __torch_dispatch__(  # ty: ignore[invalid-method-override]
        cls,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is aten.equal.default:
            return True
        return NotImplemented


def _is_alias(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def test_alias_preserves_representation_and_aliasing() -> None:
    tensor = VarLenTensor.from_list([[1.0], None, [2.0, 3.0]])
    out = aten.alias.default(tensor)

    assert isinstance(out, VarLenTensor)
    assert out.tolist() == tensor.tolist()
    assert out._base is tensor
    assert _is_alias(out, tensor)
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()
    assert out._valid is not None
    assert tensor._valid is not None
    assert out._valid.data_ptr() == tensor._valid.data_ptr()


def test_mixed_subclass_dispatch_defers_to_other_type() -> None:
    tensor = VarLenTensor.from_list([[1.0]])
    other = torch.empty(0).as_subclass(EqualTensor)

    assert aten.equal.default(tensor, other)
    assert aten.equal.default(other, tensor)
