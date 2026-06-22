import torch
from schemafm import CategoricalTensor


def test_to_copy() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor.to(torch.int32)
    assert isinstance(out, CategoricalTensor)
    assert out.dtype == torch.int32
    assert out.as_tensor().dtype == torch.int32
    assert out.categories == tensor.categories

    out = tensor.to(torch.float32)
    assert not isinstance(out, CategoricalTensor)
    assert out.dtype == torch.float32


def test_pin_memory() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    assert not tensor.is_pinned()
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()


def test_isnan() -> None:
    data = torch.tensor([[0, -1, 2], [-2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    expected = torch.tensor([[False, True, False], [True, False, False]])
    assert torch.isnan(tensor).equal(expected)
    assert tensor.isnan().equal(expected)
