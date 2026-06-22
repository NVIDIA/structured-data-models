import torch
from schemafm import CategoricalTensor


def test_isnan() -> None:
    data = torch.tensor([[0, -1, 2], [-2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    expected = torch.tensor([[False, True, False], [True, False, False]])
    assert torch.isnan(tensor).equal(expected)
    assert tensor.isnan().equal(expected)
