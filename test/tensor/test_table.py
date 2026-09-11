import io
from datetime import datetime
from textwrap import dedent
from typing import cast

import pandas as pd
import pyarrow as pa
import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NaT,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.testing import onlyCUDA, withCUDA


def test_init() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country", "segment"],
        },
        numerical=torch.randn(2, 2),
        categorical=CategoricalTensor(
            code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            categories=(torch.arange(2), torch.arange(2)),
        ),
    )
    assert repr(tensor) == dedent("""\
        TableTensor(
          size=(2, 4),
          blocks={
            numerical (2): [age, income],
            categorical (2): [country, segment],
          },
        )""")

    assert tensor.size() == (2, 4)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device("cpu")
    assert tensor.numerical.size() == (2, 2)
    assert tensor.categorical.size() == (2, 2)
    assert tensor.columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: ("country", "segment"),
        Stype.datetime: (),
        Stype.text: (),
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
        Stype.text: (),
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


def test_equal() -> None:
    numerical = torch.randn(2, 2)
    categorical = CategoricalTensor(
        code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        categories=(torch.arange(2), torch.arange(2)),
    )
    id = ColumnarTensor(
        (
            torch.arange(2),
            StringTensor.from_list(["A", "B"]),
        )
    )

    tensor1 = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country", "segment"],
            "id": ["user_id", "id"],
        },
        numerical=numerical,
        categorical=categorical,
        id=id,
    )
    tensor2 = TableTensor(
        columns={
            "numerical": ["income", "age"],
            "categorical": ["segment", "country"],
            "id": ["id", "user_id"],
        },
        numerical=numerical.flip(-1),
        categorical=cast(
            CategoricalTensor,
            torch.cat([categorical[:, 1:], categorical[:, :1]], dim=-1),
        ),
        id=cast(
            ColumnarTensor,
            torch.cat([id[:, 1:], id[:, :1]], dim=-1),
        ),
    )

    assert tensor1.equal(tensor1)
    assert tensor1.equal(tensor2)
    assert tensor1.allclose(tensor1)
    assert tensor1.allclose(tensor2)


def test_equal_categorical_categories() -> None:
    data = torch.tensor([[0], [1]], dtype=torch.int32)
    tensor1 = TableTensor(
        columns={"categorical": ["country"]},
        categorical=CategoricalTensor(
            code=data,
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )
    tensor2 = TableTensor(
        columns={"categorical": ["country"]},
        categorical=CategoricalTensor(
            code=data.clone(),
            categories=(StringTensor.from_list(["x", "y"]),),
        ),
    )

    assert tensor1.equal(tensor1.clone())
    assert tensor1.allclose(tensor1.clone())

    assert not tensor1.equal(tensor2)
    assert not tensor1.allclose(tensor2)


def test_allclose_discrete_blocks() -> None:
    datetime = torch.tensor([[1_700_000_000_000_000], [1_700_000_000_000_001]])
    tensor1 = TableTensor(
        columns={"datetime": ["time"]},
        datetime=datetime,
    )
    tensor2 = TableTensor(
        columns={"datetime": ["time"]},
        datetime=datetime + 60 * 1_000_000,  # 60 seconds later.
    )
    assert tensor1.allclose(tensor1.clone())
    assert not tensor1.allclose(tensor2)

    tensor1 = TableTensor(
        columns={"id": ["user_id"]},
        id=ColumnarTensor((torch.tensor([1_000_000, 2_000_000]),)),
    )
    tensor2 = TableTensor(
        columns={"id": ["user_id"]},
        id=ColumnarTensor((torch.tensor([1_000_001, 2_000_001]),)),
    )
    assert tensor1.allclose(tensor1.clone())
    assert not tensor1.allclose(tensor2)


def test_allclose_numerical_tolerances() -> None:
    tensor1 = TableTensor(
        columns={"numerical": ["a", "b"]},
        numerical=torch.tensor([[1.0, float("nan")]]),
    )
    tensor2 = TableTensor(
        columns={"numerical": ["a", "b"]},
        numerical=torch.tensor([[1.0 + 1e-7, float("nan")]]),
    )

    assert not tensor1.allclose(tensor2)
    assert tensor1.allclose(tensor2, equal_nan=True)
    assert not tensor1.allclose(tensor2, rtol=0.0, atol=0.0, equal_nan=True)


def test_from_tensor() -> None:
    data = torch.randn(5, 2)
    tensor = TableTensor.from_tensor(data)
    assert tensor.size() == (5, 2)
    assert tensor.columns == {
        Stype.numerical: ("0", "1"),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.text: (),
        Stype.id: (),
    }
    assert tensor.numerical.equal(data)
    assert TableTensor.from_tensor(data[:, :0]).size() == (5, 0)

    data = torch.tensor(
        [
            [0, 20],
            [-2, 10],
            [1, 10],
            [-1, 20],
        ]
    )
    tensor = TableTensor.from_tensor(data)
    assert tensor.size() == (4, 2)
    assert tensor.columns == {
        Stype.numerical: (),
        Stype.categorical: ("0", "1"),
        Stype.datetime: (),
        Stype.text: (),
        Stype.id: (),
    }
    assert tensor.categorical.code.equal(
        torch.tensor([[2, 1], [0, 0], [3, 0], [1, 1]])
    )
    assert tensor.categorical.categories[0].equal(torch.tensor([-2, -1, 0, 1]))
    assert tensor.categorical.categories[1].equal(torch.tensor([10, 20]))
    assert TableTensor.from_tensor(data[:, :0]).size() == (4, 0)

    data = StringTensor.from_list([["left", "right"], ["up", "down"]])
    tensor = TableTensor.from_tensor(data)
    assert tensor.size() == (2, 2)
    assert tensor.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.text: ("0", "1"),
        Stype.id: (),
    }
    assert tensor.text.equal(data)


def test_inference_mode() -> None:
    def make_table() -> TableTensor:
        return TableTensor(
            columns={
                "numerical": ["value"],
                "categorical": ["kind"],
                "id": ["name"],
            },
            numerical=torch.randn(3, 1),
            categorical=CategoricalTensor(
                code=torch.arange(3, dtype=torch.int32).unsqueeze(-1),
                categories=(StringTensor.from_list(["a", "b", "c"]),),
            ),
            id=ColumnarTensor((StringTensor.from_list(["x", "y", "z"]),)),
        )

    table = make_table()
    with torch.inference_mode():
        view = table[:2]

    assert not torch.is_inference(view)
    assert not torch.is_inference(view.categorical)
    assert not torch.is_inference(view.id)
    assert not torch.is_inference(view.id._columns[0])

    # Like PyTorch's NestedTensor, direct construction follows the active mode
    # while views preserve the inference state of their outer input.
    with torch.inference_mode():
        table = table.replace_blocks()
    assert torch.is_inference(table)
    assert not torch.is_inference(table.numerical)

    view = table[:2]
    assert torch.is_inference(view)
    assert not torch.is_inference(view.numerical)

    with torch.inference_mode():
        table = make_table()

    view = table[:2]
    assert torch.is_inference(view)
    assert torch.is_inference(view.categorical)
    assert torch.is_inference(view.id)
    assert torch.is_inference(view.id._columns[0])


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
            code=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["USA", "Germany"]),),
        ),
        datetime=torch.tensor([[10], [20]], dtype=torch.int64),
        id=ColumnarTensor((torch.tensor([100, 200]),)),
    )

    numerical = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    categorical = CategoricalTensor(
        code=torch.tensor([[1], [0]], dtype=torch.int32),
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
            code=torch.tensor([[0], [1]], dtype=torch.int32),
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
        Stype.text: (),
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
        Stype.text: (),
        Stype.id: (),
    }
    assert categorical.categorical is tensor.categorical

    mixed = tensor.select_stypes(["numerical", Stype.categorical])
    assert mixed.columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: ("country",),
        Stype.datetime: (),
        Stype.text: (),
        Stype.id: (),
    }
    assert mixed.numerical is tensor.numerical
    assert mixed.categorical is tensor.categorical
    assert mixed.datetime.size() == (2, 0)
    assert mixed.id.size() == (2, 0)


@withCUDA
@pytest.mark.parametrize("inference", [False, True])
def test_select_contiguous_columns_shares_storage(
    device: torch.device, inference: bool
) -> None:
    values = torch.arange(32, device=device).view(8, 4)
    tensor = TableTensor(
        columns={
            stype: tuple(f"{stype}_{i}" for i in range(4)) for stype in Stype
        },
        numerical=values.float()[::2],
        categorical=CategoricalTensor(
            code=values.remainder(4)[::2],
            categories=tuple(torch.arange(4, device=device) for _ in range(4)),
        ),
        datetime=values[::2],
        text=cast(
            StringTensor,
            StringTensor.from_list(
                [
                    [f"{row}:{column}" for column in range(4)]
                    for row in range(8)
                ],
                device=device,
            )[::2],
        ),
        id=cast(
            ColumnarTensor,
            ColumnarTensor(tuple(values[:, i] for i in range(4)))[::2],
        ),
    )

    with torch.inference_mode(inference):
        out = tensor.select_columns(
            [f"{stype}_{i}" for stype in reversed(Stype) for i in (2, 1)]
        )

    for stype, block in tensor.items():
        assert out.columns[stype] == (f"{stype}_1", f"{stype}_2")
        assert out.blocks[stype].equal(block[..., 1:3])
    for selected, original in (
        (out.numerical, tensor.numerical),
        (out.categorical.code, tensor.categorical.code),
        (out.datetime, tensor.datetime),
        (out.id[..., 0], tensor.id[..., 1]),
    ):
        assert (
            selected.untyped_storage().data_ptr()
            == original.untyped_storage().data_ptr()
        )
    out.numerical.fill_(-1)
    assert tensor.numerical[..., 1:3].eq(-1).all()


def test_drop_stypes() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
            "datetime": ["created_at"],
            "id": ["user_id"],
        },
        numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["USA", "Germany"]),),
        ),
        datetime=torch.tensor([[10], [20]], dtype=torch.int64),
        id=ColumnarTensor((torch.tensor([100, 200]),)),
    )

    no_numerical = tensor.drop_stypes(Stype.numerical)
    assert isinstance(no_numerical, TableTensor)
    assert no_numerical.columns == {
        Stype.numerical: (),
        Stype.categorical: ("country",),
        Stype.datetime: ("created_at",),
        Stype.text: (),
        Stype.id: ("user_id",),
    }
    assert no_numerical.numerical.size() == (2, 0)
    assert no_numerical.categorical is tensor.categorical
    assert no_numerical.datetime is tensor.datetime
    assert no_numerical.id is tensor.id

    no_categorical = tensor.drop_stypes(Stype.categorical)
    assert no_categorical.columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: (),
        Stype.datetime: ("created_at",),
        Stype.text: (),
        Stype.id: ("user_id",),
    }
    assert no_categorical.numerical is tensor.numerical
    assert no_categorical.categorical.size() == (2, 0)

    mixed = tensor.drop_stypes([Stype.numerical, Stype.categorical])
    assert mixed.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: ("created_at",),
        Stype.text: (),
        Stype.id: ("user_id",),
    }
    assert mixed.numerical.size() == (2, 0)
    assert mixed.categorical.size() == (2, 0)
    assert mixed.datetime is tensor.datetime
    assert mixed.id is tensor.id

    empty = tensor.drop_stypes(list(Stype))
    assert empty.size() == (2, 0)
    assert empty.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.text: (),
        Stype.id: (),
    }

    with pytest.raises(ValueError, match="not a valid Stype"):
        tensor.drop_stypes("unknown")


def test_save_load() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
            "datetime": ["created_at"],
        },
        numerical=torch.randn(3, 2),
        categorical=CategoricalTensor(
            code=torch.arange(3).view(3, 1),
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
            code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
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


@withCUDA
def test_to_dtype_preserves_empty_block_device(
    device: torch.device,
) -> None:
    tensor = TableTensor.from_tensor(
        torch.ones(2, 1, dtype=torch.float16, device=device),
    )

    with torch.inference_mode():
        out = tensor.to(torch.float32)

    assert isinstance(out, TableTensor)
    assert out.numerical.dtype == torch.float32
    assert out.device == device
    assert out.categorical.device == device
    assert out.datetime.device == device
    assert out.text.device == device
    assert out.id.device == device


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
            code=torch.randint(0, 2, (2, 3, 1), dtype=torch.int32),
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
            code=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
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

    with pytest.raises(RuntimeError, match="Can't slice"):
        _ = torch.ops.aten.slice.Tensor(tensor, -1, 0, 1, 1)
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
            code=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
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
        Stype.text: (),
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
            code=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
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
            code=torch.randint(0, 2, (2, 3, 4, 1), dtype=torch.int32),
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
        Stype.text: (),
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
        Stype.text: (),
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
            code=torch.randint(0, 2, (2, 1), dtype=torch.int32),
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
            code=torch.randint(0, 2, (3, 1), dtype=torch.int32),
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
            code=torch.randint(0, 2, (2, 1), dtype=torch.int32),
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
        Stype.text: (),
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


@withCUDA
def test_cat_all_column_empty(device: torch.device) -> None:
    tensor1 = TableTensor(size=(2,), device=device)
    tensor2 = TableTensor(size=(3,), device=device)

    out = torch.cat([tensor1, tensor2], dim=0)
    assert isinstance(out, TableTensor)
    assert out.size() == (5, 0)
    assert out.device == device
    assert out.numerical.device == device
    assert out.columns == tensor1.columns

    out = torch.cat([tensor1, tensor1], dim=-1)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 0)
    assert out.device == device
    assert out.numerical.device == device
    assert out.columns == tensor1.columns


def test_cat_stack_reorder() -> None:
    tensor1 = TableTensor(
        columns={
            "numerical": ["age", "amount"],
        },
        numerical=torch.randn(4, 2),
    )
    tensor2 = TableTensor(
        columns={
            "numerical": ["amount", "age"],
        },
        numerical=torch.randn(4, 2),
    )

    out = torch.cat([tensor1, tensor2], dim=0)
    assert isinstance(out, TableTensor)
    assert out.size() == (8, 2)
    assert out.numerical.equal(
        torch.cat([tensor1.numerical, tensor2.numerical.flip(1)], dim=0)
    )

    out = torch.stack([tensor1, tensor2], dim=0)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 4, 2)
    assert out.numerical.equal(
        torch.stack([tensor1.numerical, tensor2.numerical.flip(1)], dim=0)
    )


@withCUDA
@pytest.mark.parametrize("dim", [0, 1, -2])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize(
    ("dtype1", "dtype2"),
    [
        (torch.float32, torch.float32),
        (torch.int32, torch.int32),
        (torch.float32, torch.float64),
    ],
)
def test_stack_reorder_preserves_inputs(
    device: torch.device,
    dim: int,
    inference: bool,
    dtype1: torch.dtype,
    dtype2: torch.dtype,
) -> None:
    values = torch.arange(96, device=device).reshape(2, 8, 6)
    first = values.to(dtype1)[:, ::2, ::2]
    second = (values + 100).to(dtype2)[:, ::2, ::2]
    tensors = [
        TableTensor(
            columns={"numerical": columns},
            numerical=numerical,
            categorical=CategoricalTensor(
                code=torch.empty((2, 4, 0), dtype=torch.int64, device=device),
                categories=(),
            ),
        )
        for columns, numerical in (
            (("a", "b", "c"), first),
            (("c", "a", "b"), second),
        )
    ]
    expected = torch.stack([first, second[..., [1, 2, 0]]], dim=dim)
    originals = [first.clone(), second.clone()]

    with torch.inference_mode(inference):
        out = cast(
            TableTensor,
            torch.stack(cast(list[torch.Tensor], tensors), dim=dim),
        )
        assert out.columns == tensors[0].columns
        torch.testing.assert_close(out.numerical, expected)
        assert out.categorical.code.dtype == torch.int64
        out.numerical.fill_(-1)

    for tensor, original in zip(tensors, originals):
        torch.testing.assert_close(tensor.numerical, original)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16]
)
def test_stack_reorder_autocast(dtype: torch.dtype) -> None:
    values = torch.arange(12, dtype=dtype).reshape(4, 3)
    first = TableTensor(
        columns={"numerical": ("a", "b", "c")}, numerical=values
    )
    second = TableTensor(
        columns={"numerical": ("c", "a", "b")}, numerical=values + 20
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        if dtype == torch.float16:
            with pytest.raises(RuntimeError, match="Unexpected floating"):
                torch.stack([first, second])
        else:
            out = cast(TableTensor, torch.stack([first, second]))
            expected = torch.stack([values, second.numerical[..., [1, 2, 0]]])
            torch.testing.assert_close(out.numerical, expected)


def test_stack_reorder_rejects_mismatched_shapes() -> None:
    first = TableTensor(
        columns={"numerical": ("a", "b")}, numerical=torch.ones(4, 2)
    )
    second = TableTensor(
        columns={"numerical": ("b", "a")}, numerical=torch.ones(1, 2)
    )

    with pytest.raises(RuntimeError, match="stack expects each tensor"):
        torch.stack([first, second])


def test_stack_reorder_mixed_columns() -> None:
    values = torch.arange(12).reshape(4, 3)
    codes = torch.tensor([[0, 2], [1, 0], [0, 1], [1, 2]])
    first = TableTensor(
        columns={
            "numerical": ("a", "b", "c"),
            "categorical": ("country", "segment"),
        },
        numerical=values.float(),
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.arange(2), torch.arange(3) + 10),
        ),
    )
    second = TableTensor(
        columns={
            "numerical": ("c", "a", "b"),
            "categorical": ("segment", "country"),
        },
        numerical=(values + 20).float(),
        categorical=CategoricalTensor(
            code=codes.flip(1),
            categories=(torch.arange(3) + 10, torch.arange(2)),
        ),
    )

    out = cast(TableTensor, torch.stack([first, second]))
    assert out.columns == first.columns
    torch.testing.assert_close(
        out.numerical,
        torch.stack([first.numerical, second.numerical[:, [1, 2, 0]]]),
    )
    torch.testing.assert_close(
        out.categorical.code, torch.stack([codes, codes])
    )
    for actual, expected in zip(
        out.categorical.categories, first.categorical.categories
    ):
        torch.testing.assert_close(actual, expected)


def test_pin_memory() -> None:
    tensor = TableTensor(
        columns={"numerical": ["age", "income"]},
        numerical=torch.randn(2, 2),
    )

    assert not tensor.is_pinned()


@onlyCUDA
def test_pin_memory_cuda() -> None:
    tensor = TableTensor(
        columns={
            "numerical": ["age", "income"],
            "categorical": ["country"],
        },
        numerical=torch.randn(2, 2),
        categorical=CategoricalTensor(
            code=torch.randint(0, 2, (2, 1), dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
    )

    out = cast(TableTensor, tensor.pin_memory())
    assert out.is_pinned()
    assert out.numerical.is_pinned()
    assert out.categorical.is_pinned()
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
    assert tensor.categorical.code.equal(torch.tensor([[0], [1], [2], [0]]))
    assert tensor.categorical.categories[0].tolist() == ["US", "CA", ""]
    assert tensor.datetime.equal(
        torch.tensor(
            [
                [1704067200000000],
                [NaT],
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

    assert tensor.size() == (0, 2)

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
    assert tensor.categorical.code.equal(torch.tensor([[0, 0], [1, 1]]))
    assert tensor.categorical.categories[0].tolist() == ["US", "CA"]
    assert tensor.categorical.categories[1].tolist() == ["a", "b"]


def test_from_pandas_period() -> None:
    df = pd.DataFrame(
        {
            "year": pd.PeriodIndex(["2020", None], freq="Y"),
            "quarter": pd.PeriodIndex(["2020Q2", None], freq="Q"),
            "month": pd.PeriodIndex(["2020-02", None], freq="M"),
            "day": pd.PeriodIndex(["2020-02-03", None], freq="D"),
        }
    )

    tensor = TableTensor.from_pandas(
        df=df,
        stypes=dict.fromkeys(df.columns, Stype.datetime),
    )

    assert tensor.datetime.equal(
        torch.tensor(
            [
                [
                    1_577_836_800_000_000,
                    1_585_699_200_000_000,
                    1_580_515_200_000_000,
                    1_580_688_000_000_000,
                ],
                [NaT, NaT, NaT, NaT],
            ]
        )
    )
    assert all(isinstance(dtype, pd.PeriodDtype) for dtype in df.dtypes)


@onlyCUDA
def test_from_pandas_id_cuda() -> None:
    df = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "item_id": ["a", "b", "c"],
        }
    )

    tensor = TableTensor.from_pandas(
        df=df,
        stypes={"user_id": "id", "item_id": "id"},
        device="cuda",
    )

    assert tensor.size() == (3, 2)
    assert tensor.device.type == "cuda"
    assert tensor.id.device == tensor.device
    assert tensor.id[:, 0].equal(torch.tensor([0, 1, 2], device=tensor.device))


@onlyCUDA
def test_to_device_without_index() -> None:
    tensor = TableTensor(
        columns={"numerical": ["age"]},
        numerical=torch.randn(3, 1),
    )

    out = tensor.to("cuda")
    assert isinstance(out, TableTensor)
    assert out.device.type == "cuda"
    assert out.id.device == out.device


def test_text() -> None:
    data = {
        "age": [0.0, 1.0, 2.0],
        "title": ["hello world", "foo bar baz", "lorem ipsum dolor"],
        "body": ["a b c", "d e f", "g h i"],
    }

    tensor = TableTensor.from_arrow(
        pa.table(data),
        stypes={"age": "numerical", "title": "text", "body": "text"},
    )

    assert tensor.size() == (3, 3)
    assert tensor.columns[Stype.text] == ("title", "body")
    assert tensor.text.size() == (3, 2)

    assert tensor.to_arrow().to_pydict() == data


@onlyCUDA
def test_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    data = {
        "age": [0.0, 1.0, None, 3.0],
        "income": [10.0, None, 12.0, 13.0],
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

    tensor = TableTensor.from_cudf(
        df=cudf.DataFrame(data),
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
    assert tensor.numerical.allclose(
        torch.tensor(
            [
                [0.0, 10.0],
                [1.0, float("nan")],
                [float("nan"), 12.0],
                [3.0, 13.0],
            ],
            device=tensor.device,
        ),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert tensor.categorical.code.equal(
        torch.tensor([[0], [1], [2], [0]], device=tensor.device)
    )
    assert tensor.categorical.categories[0].tolist() == ["US", "CA", ""]
    assert tensor.datetime.equal(
        torch.tensor(
            [
                [1704067200000000],
                [NaT],
                [1704153600000000],
                [1704240000000000],
            ],
            device=tensor.device,
        )
    )
    assert tensor.id[:, 0].equal(
        torch.tensor([0, 1, 2, 3], device=tensor.device)
    )
    assert tensor.id[:, 1].equal(
        StringTensor.from_list(["a", "b", "c", "d"], device=tensor.device)
    )

    df = tensor.to_cudf()
    assert df.to_arrow().to_pydict() == data


@onlyCUDA
def test_from_cudf_empty() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = TableTensor.from_cudf(
        df=cudf.DataFrame(
            {
                "age": [],
                "country": [],
            }
        ),
        stypes={
            "age": "numerical",
            "country": "categorical",
        },
    )

    assert tensor.size() == (0, 2)
    assert tensor.is_cuda

    table = tensor.to_arrow()
    assert table.num_rows == 0
    assert table.column_names == ["age", "country"]
    assert table.to_pydict() == {"age": [], "country": []}
