import warnings
from typing import cast

import pyarrow as pa
import pytest
import torch
from sdm import CategoricalTensor, StringTensor
from sdm.testing import onlyCUDA, withCUDA


def test_to_copy_string_categories() -> None:
    data = torch.randint(0, 2, size=(10, 2))
    categories = (
        StringTensor.from_list(["USA", "GERMANY"]),
        StringTensor.from_list(["enterprise", "startup"]),
    )
    tensor = CategoricalTensor(data, categories)

    out = tensor.clone()
    assert isinstance(out, CategoricalTensor)
    for out_category, category in zip(out.categories, categories):
        assert out_category.tolist() == category.tolist()

    out = tensor.to(torch.int32)
    assert isinstance(out, CategoricalTensor)
    assert out.dtype == torch.int32
    for out_category, category in zip(out.categories, categories):
        assert out_category.tolist() == category.tolist()


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


def test_from_arrow_string_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array(["b", "a", None, "b"]),
    )

    assert tensor.equal(torch.tensor([[0], [1], [-1], [0]], dtype=torch.int32))
    assert tensor.categories[0].tolist() == ["b", "a"]


def test_from_arrow_chunked_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.chunked_array([pa.array(["b", None]), pa.array(["a", "b"])]),
    )

    assert tensor.equal(torch.tensor([[0], [-1], [1], [0]], dtype=torch.int32))
    assert tensor.categories[0].tolist() == ["b", "a"]


def test_from_arrow_numeric_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array([10, 20, None, 10], type=pa.int32()),
    )

    assert tensor.equal(torch.tensor([[0], [1], [-1], [0]], dtype=torch.int32))
    assert tensor.categories[0].equal(
        torch.tensor([10, 20], dtype=torch.int32)
    )


def test_from_arrow_all_missing_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array([None, None], type=pa.string()),
    )

    assert tensor.equal(torch.tensor([[-1], [-1]], dtype=torch.int32))
    assert tensor.categories[0].numel() == 0


def test_from_arrow_dtype() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array(["b", "a", None]),
        dtype=torch.int64,
    )

    assert tensor.dtype == torch.int64
    assert tensor.equal(torch.tensor([[0], [1], [-1]], dtype=torch.int64))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_from_arrow_does_not_warn_on_readonly_numpy(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        CategoricalTensor.from_arrow(
            pa.array([10, 20, None, 10], type=pa.int32()),
            dtype=dtype,
            device=device,
        )

    assert not any(
        "NumPy array is not writable" in str(warning.message)
        for warning in rec
    )


@onlyCUDA
def test_from_arrow_cuda() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array(["b", "a", None, "b"]),
        device="cuda",
    )

    assert tensor.device.type == "cuda"
    assert tensor.as_tensor().equal(
        torch.tensor(
            [[0], [1], [-1], [0]],
            dtype=torch.int32,
            device=tensor.device,
        )
    )
    assert tensor.categories[0].device.type == "cuda"
    assert tensor.categories[0].tolist() == ["b", "a"]

    tensor = CategoricalTensor.from_arrow(
        pa.array([10, 20, None, 10], type=pa.int32()),
        device="cuda",
    )

    assert tensor.device.type == "cuda"
    assert tensor.as_tensor().equal(
        torch.tensor(
            [[0], [1], [-1], [0]],
            dtype=torch.int32,
            device=tensor.device,
        )
    )
    assert tensor.categories[0].device.type == "cuda"
    assert tensor.categories[0].equal(
        torch.tensor([10, 20], dtype=torch.int32, device=tensor.device)
    )


def test_tolist() -> None:
    tensor = CategoricalTensor(
        data=torch.tensor(
            [
                [[0, 1], [-1, 0]],
                [[1, -1], [0, 1]],
            ],
            dtype=torch.int32,
        ),
        categories=(
            torch.tensor([10, 20]),
            torch.tensor([30, 40]),
        ),
    )

    assert tensor.tolist() == [
        [[10, 40], [None, 30]],
        [[20, None], [10, 40]],
    ]


def test_to_arrow() -> None:
    tensor = CategoricalTensor(
        data=torch.tensor(
            [
                [[0, 1], [-1, 0]],
                [[1, -1], [0, 1]],
            ],
            dtype=torch.int32,
        ),
        categories=(
            StringTensor.from_list(["US", "CA"]),
            torch.tensor([10, 20]),
        ),
    )

    table = tensor.to_arrow()
    assert table.column_names == ["0", "1"]
    assert pa.types.is_dictionary(table["0"].type)
    assert pa.types.is_dictionary(table["1"].type)
    assert table.to_pydict() == {
        "0": ["US", None, "CA", "US"],
        "1": [20, 10, None, 20],
    }


@onlyCUDA
def test_from_cudf_string_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = CategoricalTensor.from_cudf(
        cudf.Series(["b", "a", None, "b"]),
    )

    assert tensor.is_cuda
    assert tensor.as_tensor().equal(
        torch.tensor([[0], [1], [-1], [0]], device=tensor.device)
    )
    assert tensor.categories[0].is_cuda
    assert tensor.categories[0].tolist() == ["b", "a"]


@onlyCUDA
def test_from_cudf_numeric_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = CategoricalTensor.from_cudf(
        cudf.Series([10, 20, None, 10], dtype="int32"),
    )

    assert tensor.is_cuda
    assert tensor.as_tensor().equal(
        torch.tensor([[0], [1], [-1], [0]], device=tensor.device)
    )
    assert tensor.categories[0].is_cuda
    assert tensor.categories[0].tolist() == [10, 20]


@onlyCUDA
def test_from_cudf_all_missing_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = CategoricalTensor.from_cudf(
        cudf.Series([None, None], dtype="int32"),
    )

    assert tensor.is_cuda
    assert tensor.as_tensor().equal(
        torch.tensor([[-1], [-1]], device=tensor.device)
    )
    assert tensor.categories[0].is_cuda
    assert tensor.categories[0].numel() == 0


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
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
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


def test_split_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = cast(tuple[CategoricalTensor], tensor.unbind(0))
    assert all(isinstance(item, CategoricalTensor) for item in out)
    assert [item.size() for item in out] == 2 * [(3, 4)]
    assert all(item.categories == tensor.categories for item in out)

    out = tensor.unbind(-1)
    assert all(not isinstance(item, CategoricalTensor) for item in out)
    assert [item.size() for item in out] == 4 * [(2, 3)]

    out = tensor.split(2, dim=-1)
    assert all(isinstance(item, CategoricalTensor) for item in out)
    assert [item.categories for item in out] == [
        categories[:2],
        categories[2:],
    ]

    out = tensor.split([1, 3], dim=-1)
    assert all(isinstance(item, CategoricalTensor) for item in out)
    assert [item.categories for item in out] == [
        categories[:1],
        categories[1:],
    ]


def test_index_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor.index_select(1, torch.tensor([2, 0]))
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 4)
    assert out.categories == tensor.categories

    out = tensor.index_select(-1, torch.tensor([2, 0]))
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == (categories[2], categories[0])

    out = tensor[[1, 0]]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4)
    assert out.categories == tensor.categories

    out = tensor[..., [2, 0]]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == (categories[2], categories[0])

    out = tensor[..., [True, False, True, False]]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == (categories[0], categories[2])

    out = tensor[[1, 0], :, [2, 1]]
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3)

    mask = torch.zeros_like(data, dtype=torch.bool)
    mask[0, :2] = True
    out = tensor[mask]
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (8,)


def test_cat_stack() -> None:
    data1 = torch.randint(0, 4, (2, 3, 4))
    data2 = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data1.size(-1)))
    tensor1 = CategoricalTensor(data1, categories)
    tensor2 = CategoricalTensor(data2, categories)

    out = torch.cat([tensor1, tensor2], dim=0)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (4, 3, 4)
    assert out.categories == categories

    out = torch.cat([tensor1, data2], dim=0)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (4, 3, 4)

    out = torch.cat([tensor1, tensor2], dim=-1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 8)
    assert out.categories == categories + categories

    out = torch.stack([tensor1, tensor2], dim=1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 3, 4)
    assert out.categories == categories

    out = torch.stack([tensor1, data2], dim=1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 3, 4)

    out = torch.stack([tensor1, tensor2], dim=-1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4, 2)


def test_pin_memory() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    assert not tensor.is_pinned()


@onlyCUDA
def test_pin_memory_cuda() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

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
