import torch

from sdm import VarLenTensor

aten = torch.ops.aten


def _tensor() -> VarLenTensor:
    return VarLenTensor.from_list([[[0], [1, 2], [3]], [[4, 5], None, [6]]])


def test_masked_select() -> None:
    tensor = _tensor()
    mask = torch.tensor([[True, False, True], [False, True, False]])
    out = aten.masked_select.default(tensor, mask)

    assert isinstance(out, VarLenTensor)
    assert out.size() == (3,)
    assert out.stride() == (1,)
    assert out.storage_offset() == 0
    assert out.tolist() == [[0], [3], None]


def test_index_select() -> None:
    tensor = _tensor()
    out = aten.index_select.default(tensor, 0, torch.tensor([1, 0]))

    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 3)
    assert out.is_contiguous()
    assert out.tolist() == [[[4, 5], None, [6]], [[0], [1, 2], [3]]]


def test_take() -> None:
    tensor = _tensor()
    out = aten.take.default(tensor, torch.tensor([[5, 0], [4, 2]]))

    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2)
    assert out.is_contiguous()
    assert out.tolist() == [[[6], [0]], [None, [3]]]


def test_advanced_index() -> None:
    tensor = _tensor()
    out = aten.index.Tensor(
        tensor,
        [torch.tensor([1, 0]), torch.tensor([2, 1])],
    )

    assert isinstance(out, VarLenTensor)
    assert out.size() == (2,)
    assert out.is_contiguous()
    assert out.tolist() == [[6], [1, 2]]
