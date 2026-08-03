import io

import pyarrow as pa
import pytest
import torch

from sdm import CategoricalTensor, ColumnarTensor, StringTensor
from sdm.testing import onlyCUDA


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

    with pytest.raises(TypeError, match="single column tensor"):
        ColumnarTensor(
            (
                CategoricalTensor(
                    code=torch.randint(0, 2, (2, 1)),
                    categories=(torch.arange(2),),
                ),
            )
        )


def test_empty() -> None:
    tensor = ColumnarTensor((), size=(2, 3))
    assert tensor.size() == (2, 3, 0)

    with pytest.raises(ValueError, match="zero columnar data"):
        ColumnarTensor(())

    with pytest.raises(ValueError, match="to be non-empty"):
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

    tensor = ColumnarTensor.from_arrow(pa.array([1, None, 3]))
    assert tensor.tolist() == [[1], [None], [3]]
    assert tensor._validity[0] is not None
    assert tensor._validity[0].equal(torch.tensor([True, False, True]))

    tensor = ColumnarTensor.from_arrow(pa.array(["a", None, ""]))
    assert tensor.tolist() == [["a"], [None], [""]]
    assert tensor._validity[0] is not None
    assert tensor._validity[0].equal(torch.tensor([True, False, True]))

    tensor = ColumnarTensor.from_arrow(pa.array([1.5, None, 3.5]))
    assert tensor.tolist() == [[1.5], [None], [3.5]]
    assert tensor._validity[0] is not None
    assert tensor._validity[0].equal(torch.tensor([True, False, True]))


def test_from_arrow_chunked_string() -> None:
    tensor = ColumnarTensor.from_arrow(
        pa.chunked_array([pa.array(["a", "b"]), pa.array(["c"])]),
    )

    assert tensor.to_arrow().column(0).type == pa.large_string()
    assert tensor.to_arrow().to_pydict() == {"0": ["a", "b", "c"]}


@onlyCUDA
def test_from_arrow_cuda() -> None:
    tensor = ColumnarTensor.from_arrow(pa.array([1, 2, 3]), device="cuda")
    assert tensor.is_cuda
    assert tensor[:, 0].equal(torch.tensor([1, 2, 3], device=tensor.device))


@onlyCUDA
def test_from_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = ColumnarTensor.from_cudf(
        cudf.Series([1, 2, 3], dtype="int64"),
    )
    assert tensor.size() == (3, 1)
    assert tensor.is_cuda
    assert tensor[:, 0].equal(torch.tensor([1, 2, 3], device=tensor.device))

    tensor = ColumnarTensor.from_cudf(
        cudf.Series([1.5, None, 3.5], dtype="float32"),
    )
    assert tensor.size() == (3, 1)
    assert tensor.is_cuda
    assert tensor[:, 0].is_floating_point()
    assert tensor[:, 0].allclose(
        torch.tensor([1.5, float("nan"), 3.5], device=tensor.device),
        equal_nan=True,
    )

    tensor = ColumnarTensor.from_cudf(
        cudf.Series(["a", "bb", ""]),
    )
    assert tensor.size() == (3, 1)
    assert tensor.is_cuda
    assert isinstance(tensor._columns[0], StringTensor)
    assert tensor.tolist() == [["a"], ["bb"], [""]]

    tensor = ColumnarTensor.from_cudf(
        cudf.Series([1, None, 3], dtype="int64"),
    )
    assert tensor.tolist() == [[1], [None], [3]]
    assert tensor.to_cudf().to_arrow().to_pydict() == {"0": [1, None, 3]}


def test_to_arrow() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            StringTensor.from_list([["a", "b", "c"], ["d", "e", "f"]]),
        )
    )
    assert tensor.to_arrow().to_pydict() == {
        "0": [0, 1, 2, 3, 4, 5],
        "1": ["a", "b", "c", "d", "e", "f"],
    }

    tensor = ColumnarTensor(
        columns=(torch.tensor([1, 0, 3]),),
        validity=(torch.tensor([True, False, True]),),
    )
    assert tensor.to_arrow().to_pydict() == {"0": [1, None, 3]}


@onlyCUDA
def test_to_cudf() -> None:
    pytest.importorskip("cudf")

    column1 = torch.arange(6, device="cuda").view(2, 3)
    column2 = StringTensor.from_list(
        [["a", "b", "c"], ["d", "e", "f"]], device="cuda"
    )
    tensor = ColumnarTensor((column1, column2))

    df = tensor.to_cudf()
    assert df.columns.tolist() == ["0", "1"]
    assert df.to_arrow().to_pydict() == {
        "0": [0, 1, 2, 3, 4, 5],
        "1": ["a", "b", "c", "d", "e", "f"],
    }


def test_save_load() -> None:
    tensor = ColumnarTensor(
        columns=(torch.arange(3),),
        validity=(torch.tensor([True, False, True]),),
    )

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)

    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 1)
    assert out.tolist() == [[0], [None], [2]]


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
    assert out.is_cpu
    assert out.tolist() == tensor.tolist()

    with pytest.raises(TypeError, match="convert"):
        tensor.to(torch.float32)


@onlyCUDA
def test_to_cuda() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(3),
            StringTensor.from_list(["a", "bb", "c"]),
        )
    )

    out = tensor.to("cuda")
    assert isinstance(out, ColumnarTensor)
    assert out.device.type == "cuda"
    assert out.tolist() == tensor.tolist()


def test_view_ops() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            torch.arange(10, 16).view(2, 3),
        )
    )

    out = tensor.view(6, 2)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (6, 2)
    assert out.tolist() == [
        [0, 10],
        [1, 11],
        [2, 12],
        [3, 13],
        [4, 14],
        [5, 15],
    ]

    with pytest.raises(RuntimeError, match="Can't reshape"):
        _ = tensor.view(-1)

    out = tensor.unsqueeze(0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (1, 2, 3, 2)

    out = tensor.unsqueeze(0).squeeze(0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()

    with pytest.raises(RuntimeError, match="unsqueeze"):
        _ = tensor.unsqueeze(-1)

    out = tensor.unsqueeze(1).expand(-1, 4, 3, -1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 4, 3, 2)

    out = tensor.transpose(0, 1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 2, 2)

    out = tensor.permute(1, 0, 2)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 2, 2)

    with pytest.raises(RuntimeError, match="column dimension"):
        _ = tensor.permute(2, 0, 1)


def test_slicing_ops() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        )
    )

    out = tensor.select(0, 1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 4, 2)

    out = tensor.select(-1, 1)
    assert not isinstance(out, ColumnarTensor)
    assert out.equal(tensor._columns[1])

    out = tensor[:, 1:]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 4, 2)

    out = tensor[..., 1:]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 3, 4, 1)
    assert out._columns == (tensor._columns[1],)

    out = tensor.narrow(1, 1, 1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 1, 4, 2)

    rows = tensor.unbind(0)
    assert len(rows) == 2
    assert all(isinstance(row, ColumnarTensor) for row in rows)
    assert rows[0].size() == (3, 4, 2)

    columns = tensor.unbind(-1)
    assert len(columns) == 2
    assert columns[0].equal(tensor._columns[0])
    assert columns[1].equal(tensor._columns[1])

    chunks = tensor.split(1, dim=1)
    assert len(chunks) == 3
    assert all(isinstance(chunk, ColumnarTensor) for chunk in chunks)
    assert chunks[0].size() == (2, 1, 4, 2)

    chunks = tensor.split(1, dim=-1)
    assert len(chunks) == 2
    assert chunks[0].size() == (2, 3, 4, 1)
    assert chunks[0]._columns == (tensor._columns[0],)


def test_index_ops() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        )
    )

    out = tensor.index_select(1, torch.tensor([2, 0]))
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 4, 2)

    out = tensor.index_select(-1, torch.tensor([1, 0]))
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out._columns[0].equal(tensor._columns[1])
    assert out._columns[1].equal(tensor._columns[0])

    out = tensor[:, torch.tensor([2, 0])]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 4, 2)

    out = tensor[..., torch.tensor([1, 0])]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out._columns[0].equal(tensor._columns[1])
    assert out._columns[1].equal(tensor._columns[0])

    out = tensor[..., torch.tensor([True, False])]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 3, 4, 1)
    assert out._columns[0].equal(tensor._columns[0])

    mask = torch.tensor([[True, False, True], [False, True, False]])
    out = tensor[mask]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 4, 2)

    with pytest.raises(RuntimeError, match="column dimension"):
        _ = tensor[torch.tensor([0]), :, :, torch.tensor([1])]


def test_cat_stack() -> None:
    tensor1 = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            torch.arange(10, 16).view(2, 3),
        )
    )
    tensor2 = ColumnarTensor(
        (
            torch.arange(20, 26).view(2, 3),
            torch.arange(30, 36).view(2, 3),
        )
    )

    out = torch.cat([tensor1, tensor2], dim=0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (4, 3, 2)
    assert out._columns[0].equal(
        torch.cat([tensor1._columns[0], tensor2._columns[0]])
    )

    out = torch.cat([tensor1, tensor2], dim=-1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 3, 4)
    assert out._columns == (*tensor1._columns, *tensor2._columns)

    out = torch.stack([tensor1, tensor2], dim=0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 3, 2)

    with pytest.raises(RuntimeError, match="stack after"):
        _ = torch.stack([tensor1, tensor2], dim=-1)


def test_nullable_ops() -> None:
    tensor = ColumnarTensor.from_arrow(pa.array([1, None, 3]))

    out = tensor[[2, 1]]
    assert out.tolist() == [[3], [None]]

    with pytest.raises(RuntimeError, match="without losing its validity mask"):
        tensor.select(-1, 0)
    with pytest.raises(RuntimeError, match="without losing validity masks"):
        tensor.unbind(-1)

    out = torch.cat([tensor, tensor], dim=0)
    assert out.tolist() == [[1], [None], [3], [1], [None], [3]]

    other = ColumnarTensor.from_arrow(pa.array(["a", None, "c"]))
    out = torch.cat([tensor, other], dim=-1)
    assert out.tolist() == [[1, "a"], [None, None], [3, "c"]]

    out = torch.stack([tensor, tensor], dim=0)
    assert out.tolist() == [
        [[1], [None], [3]],
        [[1], [None], [3]],
    ]

    plain_string = ColumnarTensor.from_arrow(pa.array(["a", ""]))
    nullable_string = ColumnarTensor.from_arrow(pa.array(["b", None]))
    out = torch.cat([plain_string, nullable_string], dim=0)
    assert out.tolist() == [["a"], [""], ["b"], [None]]

    out = torch.stack([plain_string, nullable_string], dim=0)
    assert out.tolist() == [[["a"], [""]], [["b"], [None]]]


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


@onlyCUDA
def test_pin_memory_cuda() -> None:
    tensor = ColumnarTensor(
        (
            torch.randn(2, 3),
            torch.arange(6).view(2, 3),
        )
    )

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
