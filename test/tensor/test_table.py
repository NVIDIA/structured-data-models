import io
from datetime import datetime
from typing import cast

import pandas as pd
import pyarrow as pa
import pytest
import torch
from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)


def test_init() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country", "segment"],
        },
        numerical=torch.randn(2, 2),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            categories=(torch.arange(2), torch.arange(2)),
        ),
    )
    assert repr(tensor) == (
        "TableTensor(\n"
        "  size=(2, 4),\n"
        "  blocks={\n"
        "    numerical (2): ['age', 'income'],\n"
        "    categorical (2): ['country', 'segment'],\n"
        "    datetime (0): [],\n"
        "    id (0): [],\n"
        "  },\n"
        ")"
    )

    assert tensor.size() == (2, 4)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device("cpu")
    assert tensor.numerical.size() == (2, 2)
    assert tensor.categorical.size() == (2, 2)
    assert tensor.columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: ("country", "segment"),
        Stype.datetime: (),
        Stype.id: (),
    }
    assert tensor._column_to_loc == {
        "age": (Stype.numerical, 0),
        "income": (Stype.numerical, 1),
        "country": (Stype.categorical, 0),
        "segment": (Stype.categorical, 1),
    }

    with pytest.raises(ValueError, match=r"datetime.*dtype"):
        TableTensor(
            columns={"datetime": ["created_at"]},
            datetime=torch.zeros(2, 1),
        )


def test_empty() -> None:
    with pytest.raises(ValueError, match="to be given"):
        _ = TableTensor()
    with pytest.raises(ValueError, match="to be non-empty"):
        _ = TableTensor(())

    tensor = TableTensor(size=(1, 4))
    assert tensor.size() == (1, 4, 0)
    assert tensor.numerical.size() == (1, 4, 0)
    assert tensor.categorical.size() == (1, 4, 0)
    assert tensor.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: (),
    }
    assert tensor._column_to_loc == {}


def test_column_names() -> None:
    with pytest.raises(ValueError, match="hold 2 columns"):
        _ = TableTensor(
            columns={"numerical": ["age", "income"]},
            numerical=torch.randn(2, 3),
        )

    with pytest.raises(ValueError, match="to be unique"):
        _ = TableTensor(
            columns={"numerical": ["age", "age"]},
            numerical=torch.randn(2, 2),
        )


def test_from_tensor() -> None:
    tensor = TableTensor.from_tensor(torch.randn(5, 2))
    assert tensor.size() == (5, 2)
    assert tensor.numerical.size() == (5, 2)
    assert tensor.columns == {
        Stype.numerical: ("0", "1"),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: (),
    }


def test_save_load() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
            "datetime": ["created_at"],
        },
        numerical=torch.randn(3, 2),
        categorical=CategoricalTensor(
            data=torch.arange(3).view(3, 1),
            categories=(StringTensor.from_list(["USA, GER, FRA"]),),
        ),
        datetime=torch.tensor([[1], [2], [3]], dtype=torch.int64),
    )

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)

    assert isinstance(out, TableTensor)
    assert out.size() == tensor.size()
    assert out.numerical.equal(tensor.numerical)
    assert out.categorical.equal(tensor.categorical)
    assert out.datetime.equal(tensor.datetime)
    assert out.columns == tensor.columns
    assert out._column_to_loc == tensor._column_to_loc
    for category1, category2 in zip(
        out.categorical.categories,
        tensor.categorical.categories,
    ):
        assert category1.equal(category2)


def test_to_copy() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country", "segment"],
            "datetime": ["created_at"],
        },
        numerical=torch.randn(2, 2),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            categories=(torch.arange(2), torch.arange(2)),
        ),
        datetime=torch.tensor([[1], [2]], dtype=torch.int64),
    )

    out = tensor.to(torch.float64)

    assert isinstance(out, TableTensor)
    assert out.dtype == torch.float64
    assert out.numerical.dtype == torch.float64
    assert out.categorical.dtype == torch.int32
    assert out.datetime.equal(tensor.datetime)
    assert out.datetime.dtype == torch.int64
    assert out.columns == tensor.columns
    assert out._column_to_loc == tensor._column_to_loc


def test_clone_contiguous() -> None:
    tensor = TableTensor(
        columns={"numerical": ["age", "income"]},
        numerical=torch.randn(2, 4)[:, ::2],
    )

    out = tensor.clone()
    assert isinstance(out, TableTensor)
    assert out.numerical.equal(tensor.numerical)
    assert out.numerical.data_ptr() != tensor.numerical.data_ptr()

    assert not tensor.is_contiguous()
    assert not tensor.numerical.is_contiguous()
    out = tensor.contiguous()
    assert isinstance(out, TableTensor)
    assert out.numerical.equal(tensor.numerical)
    assert out.is_contiguous()
    assert out.numerical.is_contiguous()


def test_view_ops() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
        },
        numerical=torch.randn(2, 3, 2),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (2, 3, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )

    out = tensor.view(-1, 3)

    assert isinstance(out, TableTensor)
    assert out.size() == (6, 3)
    assert out.numerical.size() == (6, 2)
    assert out.categorical.size() == (6, 1)
    assert out.columns == tensor.columns

    with pytest.raises(RuntimeError, match="Can't reshape"):
        _ = tensor.view(-1)

    out = tensor.unsqueeze(0)
    assert isinstance(out, TableTensor)
    assert out.size() == (1, 2, 3, 3)

    out = tensor.squeeze()
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 3)

    out = tensor.unsqueeze(1).expand(-1, 4, 3, -1)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 4, 3, 3)

    out = tensor.transpose(0, 1)
    assert isinstance(out, TableTensor)
    assert out.size() == (3, 2, 3)
    assert out.numerical.size() == (3, 2, 2)
    assert out.categorical.size() == (3, 2, 1)

    out = tensor.permute(1, 0, 2)
    assert isinstance(out, TableTensor)
    assert out.size() == (3, 2, 3)
    assert out.numerical.size() == (3, 2, 2)
    assert out.categorical.size() == (3, 2, 1)


def test_slicing_ops() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
        },
        numerical=torch.randn(2, 3, 4, 2),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )

    out = tensor.select(1, 0)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 4, 3)
    assert out.numerical.size() == (2, 4, 2)
    assert out.categorical.size() == (2, 4, 1)

    out = tensor[:, :, 1:3]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 2, 3)
    assert out.numerical.size() == (2, 3, 2, 2)
    assert out.categorical.size() == (2, 3, 2, 1)

    out = tensor.narrow(-2, 1, 2)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 2, 3)
    assert out.numerical.size() == (2, 3, 2, 2)
    assert out.categorical.size() == (2, 3, 2, 1)

    with pytest.raises(RuntimeError, match="Can't select"):
        _ = tensor.select(-1, 0)
    with pytest.raises(RuntimeError, match="Can't select"):
        _ = tensor[..., 0]


def test_unbind_split() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
        },
        numerical=torch.randn(2, 3, 4, 2),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )

    out = cast(tuple[TableTensor, ...], tensor.unbind(1))
    assert len(out) == 3
    assert all(isinstance(tensor, TableTensor) for tensor in out)
    assert out[0].size() == (2, 4, 3)
    assert out[0].numerical.size() == (2, 4, 2)
    assert out[0].categorical.size() == (2, 4, 1)

    out = tensor.split(2, dim=1)
    assert len(out) == 2
    assert all(isinstance(tensor, TableTensor) for tensor in out)
    assert out[0].size() == (2, 2, 4, 3)
    assert out[1].size() == (2, 1, 4, 3)

    out = tensor.split([1, 2], dim=1)
    assert len(out) == 2
    assert all(isinstance(tensor, TableTensor) for tensor in out)
    assert out[0].size() == (2, 1, 4, 3)
    assert out[1].size() == (2, 2, 4, 3)

    out = cast(tuple[TableTensor, ...], tensor.unbind(-1))
    assert len(out) == 3
    assert all(isinstance(tensor, TableTensor) for tensor in out)
    assert out[0].size() == (2, 3, 4, 1)
    assert out[0].numerical.size() == (2, 3, 4, 1)
    assert out[0].categorical.size() == (2, 3, 4, 0)
    assert out[0].columns == {
        Stype.numerical: ("age",),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: (),
    }

    with pytest.raises(RuntimeError, match="split size 1"):
        _ = tensor.split(2, dim=-1)
    with pytest.raises(RuntimeError, match="Can't split"):
        _ = tensor.split([1, 2], dim=-1)


def test_index_ops() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
        },
        numerical=torch.randn(2, 3, 4, 2),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )

    out = tensor.index_select(1, torch.tensor([2, 0]))
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 2, 4, 3)
    assert out.numerical.size() == (2, 2, 4, 2)
    assert out.categorical.size() == (2, 2, 4, 1)

    out = tensor[:, [2, 0]]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 2, 4, 3)
    assert out.numerical.size() == (2, 2, 4, 2)
    assert out.categorical.size() == (2, 2, 4, 1)

    mask = torch.tensor([[True, False, True], [False, True, False]])
    out = tensor[mask]
    assert isinstance(out, TableTensor)
    assert out.size() == (3, 4, 3)
    assert out.numerical.size() == (3, 4, 2)
    assert out.categorical.size() == (3, 4, 1)

    with pytest.raises(RuntimeError, match="Can't index"):
        _ = tensor.index_select(-1, torch.tensor([0]))
    with pytest.raises(RuntimeError, match="Can't index"):
        _ = tensor[..., torch.tensor([0])]


def test_advanced_indexing() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
        },
        numerical=torch.randn(2, 3, 4, 2),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )

    out = tensor["age"]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 4, 1)
    assert out.numerical.size() == (2, 3, 4, 1)
    assert out.categorical.size() == (2, 3, 4, 0)
    assert out.columns == {
        Stype.numerical: ("age",),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: (),
    }

    out = cast(TableTensor, tensor.view(-1, 3))[:, "age"]
    assert isinstance(out, TableTensor)
    assert out.size() == (24, 1)
    assert out.numerical.size() == (24, 1)
    assert out.categorical.size() == (24, 0)

    out = tensor[["country", "age"]]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 4, 2)
    assert out.numerical.size() == (2, 3, 4, 1)
    assert out.categorical.size() == (2, 3, 4, 1)
    assert out.columns == {
        Stype.numerical: ("age",),
        Stype.categorical: ("country",),
        Stype.datetime: (),
        Stype.id: (),
    }

    out = tensor[..., "country"]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 4, 1)
    assert out.numerical.size() == (2, 3, 4, 0)
    assert out.categorical.size() == (2, 3, 4, 1)

    out = tensor[..., 0, ["age", "country"]]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 2)
    assert out.numerical.size() == (2, 3, 1)
    assert out.categorical.size() == (2, 3, 1)

    out = tensor[:, [2, 0], :, ["age", "country"]]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 2, 4, 2)
    assert out.numerical.size() == (2, 2, 4, 1)
    assert out.categorical.size() == (2, 2, 4, 1)

    mask = torch.tensor([[True, False, True], [False, True, False]])
    out = tensor[mask, :, "age"]
    assert isinstance(out, TableTensor)
    assert out.size() == (3, 4, 1)
    assert out.numerical.size() == (3, 4, 1)
    assert out.categorical.size() == (3, 4, 0)

    out = tensor[:, "age"]
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 3, 4, 1)
    assert out.numerical.size() == (2, 3, 4, 1)
    assert out.categorical.size() == (2, 3, 4, 0)

    with pytest.raises(IndexError, match="column dimension"):
        _ = tensor["age", :]
    with pytest.raises(KeyError):
        _ = tensor["missing"]


def test_cat_stack() -> None:
    tensor1 = TableTensor(
        columns={
            "numerical": ["age"],
            "categorical": ["country"],
        },
        numerical=torch.randn(2, 1),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (2, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )
    tensor2 = TableTensor(
        columns={
            "numerical": ["age"],
            "categorical": ["country"],
        },
        numerical=torch.randn(3, 1),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (3, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )
    tensor3 = TableTensor(
        columns={
            "numerical": ["income"],
            "categorical": ["segment"],
        },
        numerical=torch.randn(2, 1),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (2, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )

    out = torch.cat([tensor1, tensor2], dim=0)
    assert isinstance(out, TableTensor)
    assert out.size() == (5, 2)
    assert out.numerical.size() == (5, 1)
    assert out.categorical.size() == (5, 1)
    assert out.columns == tensor1.columns

    out = torch.cat([tensor1, tensor3], dim=-1)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 4)
    assert out.numerical.size() == (2, 2)
    assert out.categorical.size() == (2, 2)
    assert out.columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: ("country", "segment"),
        Stype.datetime: (),
        Stype.id: (),
    }

    out = torch.stack([tensor1, tensor1], dim=0)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 2, 2)
    assert out.numerical.size() == (2, 2, 1)
    assert out.categorical.size() == (2, 2, 1)
    assert out.columns == tensor1.columns

    with pytest.raises(RuntimeError, match="Can't stack"):
        _ = torch.stack([tensor1, tensor1], dim=-1)


def test_pin_memory() -> None:
    tensor = TableTensor(
        columns={"numerical": ["age", "income"]},
        numerical=torch.randn(2, 2),
    )

    assert not tensor.is_pinned()
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()


def test_share_memory() -> None:
    tensor = TableTensor(
        numerical=torch.randn(3, 2),
        columns={"numerical": ["age", "income"]},
    )

    assert not tensor.is_shared()
    try:
        tensor.share_memory_()
        assert tensor.is_shared()
    except RuntimeError:
        pass


def test_arrow() -> None:
    data = {
        "age": [0.0, 1.0, 2.0, 3.0],
        "income": [10.0, 11.0, 12.0, 13.0],
        "country": ["US", "CA", "", "US"],
        "time": [
            datetime(2024, 1, 1, 0, 0),
            None,
            datetime(2024, 1, 2, 0, 0),
            datetime(2024, 1, 3, 0, 0),
        ],
        "user_id": [0, 1, 2, 3],
        "item_id": ["a", "b", "c", "d"],
    }

    tensor = TableTensor.from_arrow(
        pa.table(data),
        stypes={
            "age": "numerical",
            "income": "numerical",
            "country": "categorical",
            "time": "datetime",
            "user_id": "id",
            "item_id": "id",
        },
    )

    assert tensor.size() == (4, 6)
    assert tensor.numerical.equal(
        torch.tensor(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
            ]
        )
    )
    assert tensor.categorical.equal(torch.tensor([[0], [1], [2], [0]]))
    assert tensor.categorical.categories[0].tolist() == ["US", "CA", ""]
    assert tensor.datetime.equal(
        torch.tensor(
            [
                [1704067200000000],
                [-9223372036854775808],
                [1704153600000000],
                [1704240000000000],
            ]
        )
    )
    assert tensor.id[:, 0].equal(torch.tensor([0, 1, 2, 3]))
    assert tensor.id[:, 1].equal(StringTensor.from_list(["a", "b", "c", "d"]))

    assert tensor.to_arrow().to_pydict() == data


def test_arrow_empty() -> None:
    tensor = TableTensor.from_arrow(
        pa.table(
            {
                "age": pa.array([], type=pa.float32()),
                "country": pa.array([], type=pa.string()),
            }
        ),
        stypes={
            "age": "numerical",
            "country": "categorical",
        },
    )

    table = tensor.to_arrow()

    assert table.num_rows == 0
    assert table.column_names == ["age", "country"]
    assert table.to_pydict() == {"age": [], "country": []}


def test_from_pandas() -> None:
    df = pd.DataFrame(
        {
            "age": [10, 20],
            "income": [1.0, 2.5],
            "country": ["US", "CA"],
            "segment": ["a", "b"],
        }
    )

    tensor = TableTensor.from_pandas(
        df=df,
        stypes={
            "age": "numerical",
            "income": "numerical",
            "country": "categorical",
            "segment": "categorical",
        },
    )

    assert tensor.size() == (2, 4)
    assert tensor.numerical.equal(torch.tensor([[10.0, 1.0], [20.0, 2.5]]))
    assert tensor.categorical.equal(torch.tensor([[0, 0], [1, 1]]))
    assert tensor.categorical.categories[0].tolist() == ["US", "CA"]
    assert tensor.categorical.categories[1].tolist() == ["a", "b"]
