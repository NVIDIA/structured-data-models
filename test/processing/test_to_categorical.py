import pyarrow as pa
import pytest
import torch
from sdm import Stype, TableTensor
from sdm.processing import ToCategorical
from sdm.testing import onlyCUDA


def _table() -> TableTensor:
    return TableTensor.from_arrow(
        pa.table(
            {
                "kind": pa.array(["x", "y", "x"]),
                "bio": pa.array(["a b", "c d", "e f"]),
            }
        ),
        stypes={"kind": "categorical", "bio": "text"},
    )


def test_to_categorical_converts_text_stype() -> None:
    table = _table()

    output = ToCategorical().transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.categorical] == ("kind", "bio")
    assert output.columns[Stype.text] == ()
    assert output.text.size(-1) == 0
    assert torch.equal(
        output.categorical.as_tensor()[..., 0],
        table.categorical.as_tensor()[..., 0],
    )
    assert output.categorical.as_tensor()[..., 1].tolist() == [0, 1, 2]
    assert output.categorical.categories[1].tolist() == ["a b", "c d", "e f"]


def test_to_categorical_is_identity_for_table_without_text() -> None:
    table = _table().select_columns(("kind",))

    assert ToCategorical().transform(table) is table


def test_to_categorical_rejects_unsupported_stype() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]),
        columns=("value",),
    )

    with pytest.raises(ValueError, match="numerical"):
        ToCategorical().transform(table)


@onlyCUDA
def test_to_categorical_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    df = cudf.DataFrame({"bio": ["a b", "c d", "a b"]})
    table = TableTensor.from_cudf(df, stypes={"bio": "text"})

    output = ToCategorical().transform(table)

    assert output.columns[Stype.categorical] == ("bio",)
    assert output.categorical.is_cuda
    assert output.categorical.as_tensor().view(-1).tolist() == [0, 1, 0]
    assert output.categorical.categories[0].tolist() == ["a b", "c d"]
