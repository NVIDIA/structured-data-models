import pytest
import torch
from schemafm import CategoricalTensor, Stype, TableTensor


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

    assert tensor.size() == (2, 4)
    with pytest.raises(RuntimeError, match="single dtype"):
        _ = tensor.dtype
    assert tensor.device == torch.device("cpu")
    assert tensor._numerical.size() == (2, 2)
    assert tensor._categorical.size() == (2, 2)
    assert tensor._columns == {
        Stype.numerical: ("age", "income"),
        Stype.categorical: ("country", "segment"),
    }
    assert tensor._column_to_loc == {
        "age": (Stype.numerical, 0),
        "income": (Stype.numerical, 1),
        "country": (Stype.categorical, 0),
        "segment": (Stype.categorical, 1),
    }


def test_empty() -> None:
    with pytest.raises(ValueError, match="to be given"):
        _ = TableTensor()
    with pytest.raises(ValueError, match="to be non-empty"):
        _ = TableTensor(size=())

    tensor = TableTensor(size=(1, 4))
    assert tensor.size() == (1, 4, 0)
    assert tensor._numerical.size() == (1, 4, 0)
    assert tensor._categorical.size() == (1, 4, 0)
    assert tensor._columns == {
        Stype.numerical: (),
        Stype.categorical: (),
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
