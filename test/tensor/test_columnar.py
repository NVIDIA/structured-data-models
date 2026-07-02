import io

import pyarrow as pa
import pytest
import torch
from sdm import CategoricalTensor, ColumnarTensor, StringTensor, TableTensor


def test_init() -> None:
    value1 = torch.arange(6).view(2, 3)
    value2 = torch.randn(2, 3)

    tensor = ColumnarTensor((value1, value2))
    assert tensor.size() == (2, 3, 2)
    assert tensor.device == value1.device == value2.device
    assert repr(tensor) == "ColumnarTensor(size=(2, 3, 2))"

    with pytest.raises(ValueError, match="to have size"):
        ColumnarTensor((torch.ones(2),), size=(3,))

    with pytest.raises(ValueError, match="to have size"):
        ColumnarTensor((torch.ones(2), torch.ones(3)))

    value1 = torch.ones(2)
    value2 = CategoricalTensor(
        data=torch.randint(0, 2, (2,)),
        categories=(torch.arange(2), torch.arange(2)),
    )
    with pytest.raises(ValueError, match="hold a single column"):
        ColumnarTensor((value1, value2))


def test_empty() -> None:
    tensor = ColumnarTensor((), size=(2, 3))
    assert tensor.size() == (2, 3, 0)

    with pytest.raises(ValueError, match="zero columnar data"):
        ColumnarTensor(())

    with pytest.raises(ValueError, match="at least one dimension"):
        ColumnarTensor((), size=())


def test_from_arrow() -> None:
    tensor = ColumnarTensor.from_arrow(pa.array([1, 2, 3]))
    assert tensor.size() == (3, 1)
    assert isinstance(tensor._columns[0], torch.Tensor)
    assert not isinstance(tensor._columns[0], StringTensor)
    assert tensor.tolist() == [[1], [2], [3]]

    tensor = ColumnarTensor.from_arrow(pa.array(["a", "bb", ""]))
    assert tensor.size() == (3, 1)
    assert isinstance(tensor._columns[0], StringTensor)
    assert tensor.tolist() == [["a"], ["bb"], [""]]

    with pytest.raises(ValueError, match="cannot represent null values"):
        ColumnarTensor.from_arrow(pa.array([1, None, 3]))

    table = TableTensor.from_arrow(
        pa.table(
            {
                "user_id": pa.array([1, 2]),
                "org_id": pa.array(["a", "b"]),
                "x": pa.array([0.1, 0.2]),
            }
        ),
        stypes={
            "user_id": "id",
            "org_id": "id",
            "x": "numerical",
        },
    )
    assert table.id.size() == (2, 2)
    assert table.id.tolist() == [[1, "a"], [2, "b"]]


def test_save_load() -> None:
    tensor = ColumnarTensor((torch.arange(3),))

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)

    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 1)


def test_to_copy() -> None:
    column1 = torch.arange(12).view(2, 3, 2)[..., 0]
    column2 = torch.randn(2, 3, 2)[..., 1]
    tensor = ColumnarTensor((column1, column2))

    assert not tensor.is_contiguous()

    out = tensor.contiguous()
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out.is_contiguous()
    assert out.tolist() == tensor.tolist()

    out = tensor.clone()
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out.tolist() == tensor.tolist()

    out = tensor.to("cpu")
    assert isinstance(out, ColumnarTensor)
    assert out.device.type == "cpu"
    assert out.tolist() == tensor.tolist()

    with pytest.raises(TypeError, match="convert"):
        tensor.to(torch.float32)


def test_tolist() -> None:
    tensor = ColumnarTensor(
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[10, 20], [30, 40]]),
        )
    )

    assert tensor.tolist() == [
        [[1, 10], [2, 20]],
        [[3, 30], [4, 40]],
    ]


def test_pin_memory() -> None:
    tensor = ColumnarTensor(
        (
            torch.randn(2, 3),
            torch.arange(6).view(2, 3),
        )
    )

    assert not tensor.is_pinned()
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()


def test_share_memory() -> None:
    tensor = ColumnarTensor(
        (
            torch.randn(2, 3),
            torch.arange(6).view(2, 3),
        )
    )

    assert not tensor.is_shared()
    try:
        tensor.share_memory_()
        assert tensor.is_shared()
    except RuntimeError:
        pass
