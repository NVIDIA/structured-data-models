from collections.abc import Callable
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
import torch
from sdm import Stype, TableTensor
from sdm.testing import onlyCUDA, withCUDA

_DATA = {
    "age": [10, 20, None, 40],
    "income": [1.0, 2.5, 3.5, None],
    "country": ["US", "CA", None, "US"],
    "segment": ["a", "b", "a", None],
}
_STYPES = {
    "age": Stype.numerical,
    "income": Stype.numerical,
    "country": Stype.categorical,
    "segment": Stype.categorical,
}
_EXPECTED_COLUMNS = {
    Stype.numerical: ("age", "income"),
    Stype.categorical: ("country", "segment"),
}
_EXPECTED_CATEGORICAL = torch.tensor(
    [
        [0, 0],
        [1, 1],
        [-1, 0],
        [0, -1],
    ],
    dtype=torch.int32,
)


def _df(data: dict[str, list[Any]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(_DATA if data is None else data)


def _arrow_table(data: dict[str, list[Any]] | None = None) -> pa.Table:
    data = _DATA if data is None else data
    return pa.table(
        {
            "age": pa.array(data["age"], type=pa.int64()),
            "income": pa.array(data["income"], type=pa.float64()),
            "country": pa.array(data["country"]),
            "segment": pa.array(data["segment"]),
        }
    )


def _assert_table_tensor(
    tensor: TableTensor,
    *,
    device: torch.device,
) -> None:
    assert tensor.size() == (4, 4)
    assert tensor.device == device
    assert tensor.columns == _EXPECTED_COLUMNS
    assert tensor.numerical.dtype == torch.float32
    assert torch.allclose(
        tensor.numerical[:, 0].cpu(),
        torch.tensor([10.0, 20.0, float("nan"), 40.0]),
        equal_nan=True,
    )
    assert tensor.categorical.as_tensor().cpu().equal(_EXPECTED_CATEGORICAL)


@withCUDA
@pytest.mark.parametrize(
    "make_tensor",
    [
        pytest.param(
            lambda device: TableTensor.from_pandas(
                _df(),
                _STYPES,
                device=device,
            ),
            id="pandas",
        ),
        pytest.param(
            lambda device: TableTensor.from_arrow(
                _arrow_table(),
                _STYPES,
                device=device,
            ),
            id="arrow",
        ),
    ],
)
def test_from_table_backend(
    make_tensor: Callable[[torch.device], TableTensor],
    device: torch.device,
) -> None:
    _assert_table_tensor(make_tensor(device), device=device)


@withCUDA
@pytest.mark.parametrize(
    "make_tensor",
    [
        pytest.param(
            lambda device: TableTensor.from_pandas(
                _df(),
                _STYPES,
                device=device,
            ),
            id="pandas",
        ),
        pytest.param(
            lambda device: TableTensor.from_arrow(
                _arrow_table(),
                _STYPES,
                device=device,
            ),
            id="arrow",
        ),
    ],
)
def test_from_table_preserves_string_categories(
    make_tensor: Callable[[torch.device], TableTensor],
    device: torch.device,
) -> None:
    tensor = make_tensor(device)

    assert tensor.categorical.categories[0].tolist() == ["US", "CA"]
    assert tensor.categorical.categories[1].tolist() == ["a", "b"]


@withCUDA
def test_from_arrow_mapping(device: torch.device) -> None:
    tensor = TableTensor.from_arrow(
        {
            "age": pa.array([10, 20], type=pa.int64()),
            "income": pa.array([1.0, 2.5], type=pa.float64()),
            "country": pa.array(["US", "CA"]),
            "segment": pa.array(["a", "b"]),
        },
        _STYPES,
        device=device,
    )

    assert tensor.size() == (2, 4)
    assert tensor.device == device
    assert (
        tensor.categorical.as_tensor()
        .cpu()
        .equal(torch.tensor([[0, 0], [1, 1]], dtype=torch.int32))
    )


@withCUDA
@pytest.mark.parametrize(
    "make_tensor",
    [
        pytest.param(
            lambda device: TableTensor.from_pandas(
                _df(
                    {
                        "age": [1],
                        "income": [2],
                        "country": ["MX"],
                        "segment": ["z"],
                    }
                ),
                _STYPES,
                device=device,
            ),
            id="pandas",
        ),
        pytest.param(
            lambda device: TableTensor.from_arrow(
                _arrow_table(
                    {
                        "age": [1],
                        "income": [2],
                        "country": ["MX"],
                        "segment": ["z"],
                    }
                ),
                _STYPES,
                device=device,
            ),
            id="arrow",
        ),
    ],
)
def test_categories_are_local_to_each_input(
    make_tensor: Callable[[torch.device], TableTensor],
    device: torch.device,
) -> None:
    tensor = make_tensor(device)

    assert (
        tensor.categorical.as_tensor()
        .cpu()
        .equal(torch.tensor([[0, 0]], dtype=torch.int32))
    )


def test_tensor_frame_bridge_shape() -> None:
    torch_frame = pytest.importorskip("torch_frame")

    tensor = TableTensor.from_pandas(_df(), _STYPES)

    tf = torch_frame.TensorFrame(
        feat_dict={
            Stype.numerical: tensor.numerical,
            Stype.categorical: tensor.categorical.as_tensor(),
        },
        col_names_dict={
            Stype.numerical: list(tensor.columns[Stype.numerical]),
            Stype.categorical: list(tensor.columns[Stype.categorical]),
        },
        num_rows=tensor.size(0),
    )

    assert tf.num_rows == 4
    assert tf.feat_dict[Stype.numerical].shape == (4, 2)
    assert tf.feat_dict[Stype.categorical].shape == (4, 2)


@onlyCUDA
@pytest.mark.parametrize(
    "device",
    ["cuda", torch.device("cuda")],
    ids=["str", "device"],
)
def test_from_table_accepts_non_indexed_cuda_device(
    device: torch.device | str,
) -> None:
    # A non-indexed device such as "cuda" must resolve to the concrete device
    # of the materialized blocks (e.g. "cuda:0") instead of being rejected.
    for tensor in (
        TableTensor.from_pandas(_df(), _STYPES, device=device),
        TableTensor.from_arrow(_arrow_table(), _STYPES, device=device),
    ):
        assert tensor.device.type == "cuda"
        assert tensor.numerical.device.type == "cuda"
        assert tensor.categorical.as_tensor().device.type == "cuda"
