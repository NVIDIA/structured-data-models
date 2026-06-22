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
    for out_category, category in zip(out.categories, tensor.categories):
        assert out_category.equal(category)
        assert out_category.data_ptr() != category.data_ptr()

    out = tensor.to(torch.float32)
    assert not isinstance(out, CategoricalTensor)
    assert out.dtype == torch.float32


def test_view_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor.view(6, 4)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (6, 4)

    out = tensor.view(-1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (24,)

    out = tensor.unsqueeze(1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 1, 3, 4)

    out = tensor.unsqueeze(1).squeeze(1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4)

    out = tensor.unsqueeze(-1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4, 1)

    out = tensor.transpose(0, 1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (3, 2, 4)

    out = tensor.permute(2, 0, 1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (4, 2, 3)


def test_slicing_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.full((2,), i) for i in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor[:, 1:]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 4)
    assert out.categories == tensor.categories

    out = tensor[..., 1::2]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == categories[1::2]

    out = tensor.narrow(-1, 1, 2)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == categories[1:3]

    out = tensor[0]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (3, 4)
    assert out.categories == tensor.categories

    out = tensor[..., 0]
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3)


def test_pin_memory() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    assert not tensor.is_pinned()
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()


def test_share_memory() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    assert not tensor.is_shared()
    try:
        tensor.share_memory_()
        assert tensor.is_shared()
    except RuntimeError:
        pass


def test_isnan() -> None:
    data = torch.tensor([[0, -1, 2], [-2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    expected = torch.tensor([[False, True, False], [True, False, False]])
    assert torch.isnan(tensor).equal(expected)
    assert tensor.isnan().equal(expected)
