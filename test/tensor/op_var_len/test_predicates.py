import torch

from sdm import VarLenTensor

aten = torch.ops.aten


def test_isnan_and_isfinite_materialize_boolean_tensors() -> None:
    tensor = VarLenTensor.from_list([[1.0], None, [float("nan")]])

    isnan = aten.isnan.default(tensor)
    isfinite = torch.isfinite(tensor)
    method_isfinite = tensor.isfinite()

    assert type(isnan) is torch.Tensor
    assert type(isfinite) is torch.Tensor
    assert type(method_isfinite) is torch.Tensor
    assert isnan.equal(torch.tensor([False, True, False]))
    assert isfinite.equal(torch.tensor([True, False, True]))
    assert method_isfinite.equal(isfinite)

    isfinite[0] = False
    assert tensor.valid is not None
    assert tensor.valid.equal(torch.tensor([True, False, True]))


def test_non_nullable_predicates() -> None:
    tensor = VarLenTensor.from_list([[1], [], [2]])

    assert aten.isnan.default(tensor).equal(torch.zeros(3, dtype=torch.bool))
    assert torch.isfinite(tensor).equal(torch.ones(3, dtype=torch.bool))


def test_equal_compares_shape_validity_offsets_and_data() -> None:
    tensor = VarLenTensor.from_list([[1], None, [2, 3]])
    same = VarLenTensor.from_list([[1], None, [2, 3]])
    different_valid = VarLenTensor.from_list([[1], [], [2, 3]])
    different_offset = VarLenTensor.from_list([[1, 2], None, [3]])
    different_data = VarLenTensor.from_list([[1], None, [2, 4]])

    assert aten.equal.default(tensor, same)
    assert not aten.equal.default(tensor, different_valid)
    assert not aten.equal.default(tensor, different_offset)
    assert not aten.equal.default(tensor, different_data)
    assert not aten.equal.default(tensor, torch.arange(3))


def test_allclose_honors_tolerances_and_equal_nan() -> None:
    tensor = VarLenTensor.from_list([[1.0], [float("nan"), 2.0]])
    close = VarLenTensor.from_list([[1.001], [float("nan"), 2.001]])

    assert not aten.allclose.default(tensor, close)
    assert aten.allclose.default(
        tensor,
        close,
        0.0,
        0.01,
        True,
    )
    assert not aten.allclose.default(tensor, torch.arange(2))
