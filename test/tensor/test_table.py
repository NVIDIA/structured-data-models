import io
from datetime import datetime
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest
import torch
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.testing import withCUDA


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


def test_replace_blocks() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
            "datetime": ["created_at"],
            "id": ["user_id"],
        },
        numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["USA", "Germany"]),),
        ),
        datetime=torch.tensor([[10], [20]], dtype=torch.int64),
        id=ColumnarTensor((torch.tensor([100, 200]),)),
    )

    numerical = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    categorical = CategoricalTensor(
        data=torch.tensor([[1], [0]], dtype=torch.int32),
        categories=(StringTensor.from_list(["France", "Spain"]),),
    )
    datetime = torch.tensor([[30], [40]], dtype=torch.int64)
    id = ColumnarTensor((torch.tensor([300, 400]),))

    out = tensor.replace_blocks(
        numerical=numerical,
        categorical=categorical,
        datetime=datetime,
        id=id,
    )

    assert isinstance(out, TableTensor)
    assert out is not tensor
    assert out.columns == tensor.columns
    assert out.numerical is numerical
    assert out.categorical is categorical
    assert out.datetime is datetime
    assert out.id is id

    numerical_only = tensor.replace_blocks(numerical=numerical)
    assert numerical_only.numerical is numerical
    assert numerical_only.categorical is tensor.categorical
    assert numerical_only.datetime is tensor.datetime
    assert numerical_only.id is tensor.id


def test_replace_blocks_validates_replacement_shape() -> None:
    tensor = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        columns=["age", "income"],
    )

    with pytest.raises(ValueError, match="hold 2 columns"):
        tensor.replace_blocks(numerical=torch.ones(2, 3))


def test_select_stypes() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
            "datetime": ["created_at"],
            "id": ["user_id"],
        },
        numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["USA", "Germany"]),),
        ),
        datetime=torch.tensor([[10], [20]], dtype=torch.int64),
        id=ColumnarTensor((torch.tensor([100, 200]),)),
    )

    numerical = tensor.select_stypes(Stype.numerical)
    assert isinstance(numerical, TableTensor)
    assert numerical.columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: (),
    }
    assert numerical.numerical is tensor.numerical
    assert numerical.categorical.size() == (2, 0)
    assert numerical.datetime.size() == (2, 0)
    assert numerical.id.size() == (2, 0)

    categorical = tensor.select_stypes("categorical")
    assert categorical.columns == {
        Stype.numerical: (),
        Stype.categorical: ("country",),
        Stype.datetime: (),
        Stype.id: (),
    }
    assert categorical.categorical is tensor.categorical

    mixed = tensor.select_stypes(["numerical", Stype.categorical])
    assert mixed.columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: ("country",),
        Stype.datetime: (),
        Stype.id: (),
    }
    assert mixed.numerical is tensor.numerical
    assert mixed.categorical is tensor.categorical
    assert mixed.datetime.size() == (2, 0)
    assert mixed.id.size() == (2, 0)


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
        out = cast(TableTensor, tensor.pin_memory())
        assert out.is_pinned()
        assert out.numerical.is_pinned()
        assert out.categorical is tensor.categorical
        assert out.datetime is tensor.datetime
        assert out.id is tensor.id


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


_CUDF_DATA = {
    "age": [10, 20, None, 40],
    "income": [1.0, 2.5, 3.5, None],
    "country": ["US", "CA", None, "US"],
    "segment": ["a", "b", "a", None],
}
_CUDF_STYPES = {
    "age": Stype.numerical,
    "income": Stype.numerical,
    "country": Stype.categorical,
    "segment": Stype.categorical,
}
_CUDF_EXPECTED_COLUMNS = {
    Stype.numerical: ("age", "income"),
    Stype.categorical: ("country", "segment"),
    Stype.datetime: (),
    Stype.id: (),
}
_CUDF_EXPECTED_CATEGORICAL = torch.tensor(
    [
        [0, 0],
        [1, 1],
        [-1, 0],
        [0, -1],
    ],
    dtype=torch.int32,
)


def _import_cudf() -> Any:
    cudf = pytest.importorskip("cudf")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return cudf


def _cudf_dataframe(data: dict[str, list[Any]] | None = None) -> Any:
    cudf = _import_cudf()
    return cudf.DataFrame(_CUDF_DATA if data is None else data)


def _assert_cudf_table_tensor(
    tensor: TableTensor,
    device: torch.device,
) -> None:
    assert tensor.size() == (4, 4)
    assert tensor.device == device
    assert tensor.columns == _CUDF_EXPECTED_COLUMNS

    assert tensor.numerical.dtype == torch.float32
    assert tensor.numerical.device == device
    assert torch.allclose(
        tensor.numerical[:, 0],
        torch.tensor([10.0, 20.0, float("nan"), 40.0], device=device),
        equal_nan=True,
    )
    assert torch.allclose(
        tensor.numerical[:, 1],
        torch.tensor([1.0, 2.5, 3.5, float("nan")], device=device),
        equal_nan=True,
    )

    assert tensor.categorical.as_tensor().device == device
    assert tensor.categorical.as_tensor().equal(
        _CUDF_EXPECTED_CATEGORICAL.to(device)
    )
    assert tensor.categorical.categories[0].device == device
    assert tensor.categorical.categories[1].device == device
    assert tensor.categorical.categories[0].tolist() == ["US", "CA"]
    assert tensor.categorical.categories[1].tolist() == ["a", "b"]


@withCUDA
def test_from_cudf(device: torch.device) -> None:
    tensor = TableTensor.from_cudf(
        df=_cudf_dataframe(),
        stypes=_CUDF_STYPES,
        device=device,
    )

    _assert_cudf_table_tensor(tensor, device)


def test_from_cudf_defaults_to_cuda() -> None:
    tensor = TableTensor.from_cudf(
        df=_cudf_dataframe(),
        stypes=_CUDF_STYPES,
    )

    assert tensor.device.type == "cuda"
    _assert_cudf_table_tensor(tensor, tensor.device)


@withCUDA
def test_from_cudf_numerical_uses_cudf_dlpack(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cudf = _import_cudf()

    def fail_to_cupy(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("numerical cuDF ingestion should not use CuPy")

    monkeypatch.setattr(cudf.Series, "to_cupy", fail_to_cupy)

    tensor = TableTensor.from_cudf(
        df=cudf.DataFrame(
            {
                "age": [10, None, 30],
                "income": [1.5, None, 3.5],
            }
        ),
        stypes={
            "age": Stype.numerical,
            "income": Stype.numerical,
        },
        device=device,
    )

    assert tensor.numerical.dtype == torch.float32
    assert tensor.numerical.device == device
    assert torch.allclose(
        tensor.numerical,
        torch.tensor(
            [
                [10.0, 1.5],
                [float("nan"), float("nan")],
                [30.0, 3.5],
            ],
            device=device,
        ),
        equal_nan=True,
    )


@withCUDA
def test_from_cudf_datetime_stype(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cudf = _import_cudf()

    def fail_to_cupy(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("datetime cuDF ingestion should not use CuPy")

    monkeypatch.setattr(cudf.Series, "to_cupy", fail_to_cupy)

    tensor = TableTensor.from_cudf(
        df=cudf.DataFrame(
            {
                "created_at": cudf.Series(
                    ["2024-01-01", None, "2024-01-03"],
                    dtype="datetime64[ns]",
                ),
            }
        ),
        stypes={
            "created_at": Stype.datetime,
        },
        device=device,
    )

    assert tensor.size() == (3, 1)
    assert tensor.device == device
    assert tensor.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: ("created_at",),
        Stype.id: (),
    }
    assert tensor.datetime.device == device
    assert tensor.datetime.equal(
        torch.tensor(
            [
                [1704067200000000],
                [torch.iinfo(torch.int64).min],
                [1704240000000000],
            ],
            device=device,
        )
    )


@withCUDA
def test_from_cudf_id_stype(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cudf = _import_cudf()

    def fail_to_cupy(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("id cuDF ingestion should not use CuPy")

    monkeypatch.setattr(cudf.Series, "to_cupy", fail_to_cupy)

    tensor = TableTensor.from_cudf(
        df=cudf.DataFrame(
            {
                "user_id": cudf.Series([10, 20, 30], dtype="int64"),
                "item_id": cudf.Series(["a", "bb", ""]),
            }
        ),
        stypes={
            "user_id": Stype.id,
            "item_id": Stype.id,
        },
        device=device,
    )

    assert tensor.size() == (3, 2)
    assert tensor.device == device
    assert tensor.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: ("user_id", "item_id"),
    }
    assert tensor.id.device == device
    assert tensor.id[:, 0].equal(torch.tensor([10, 20, 30], device=device))
    assert tensor.id[:, 1].equal(
        StringTensor.from_list(["a", "bb", ""], device=device)
    )


@withCUDA
def test_from_cudf_rejects_null_integer_id(device: torch.device) -> None:
    cudf = _import_cudf()

    with pytest.raises(ValueError, match="cannot represent null integer"):
        TableTensor.from_cudf(
            df=cudf.DataFrame(
                {
                    "user_id": cudf.Series(
                        [10, None, 30],
                        dtype="int64",
                    ),
                }
            ),
            stypes={"user_id": Stype.id},
            device=device,
        )


@withCUDA
def test_from_cudf_empty_blocks_use_target_device(
    device: torch.device,
) -> None:
    numerical = TableTensor.from_cudf(
        df=_cudf_dataframe(),
        stypes={"age": Stype.numerical},
        device=device,
    )
    categorical = TableTensor.from_cudf(
        df=_cudf_dataframe(),
        stypes={"country": Stype.categorical},
        device=device,
    )

    assert numerical.device == device
    assert numerical.numerical.device == device
    assert numerical.categorical.as_tensor().device == device
    assert categorical.device == device
    assert categorical.numerical.device == device
    assert categorical.categorical.as_tensor().device == device
