import pyarrow as pa
import pytest
import torch

from sdm import ColumnarTensor, TableTensor
from sdm.relational.join import join_index
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_join_index(dtype: torch.dtype, device: torch.device) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(8, device=device),)),
    )

    left_index, right_index = join_index(
        left_table=left_table,
        right_table=right_table,
        left_keys=["id"],
        right_keys=["id"],
        dtype=dtype,
    )

    assert left_index.dtype == dtype
    assert right_index.dtype == dtype
    assert left_index.device == device
    assert right_index.device == device
    assert left_index.sort()[0].equal(torch.arange(8, device=device))
    assert right_index.equal(left_index)


@withCUDA
def test_join_index_ignores_null_keys(device: torch.device) -> None:
    left_table = TableTensor.from_arrow(
        table=pa.table({"id": [0, None, 3, 99]}),
        stypes={"id": "id"},
        device=device,
    )
    right_table = TableTensor.from_arrow(
        table=pa.table({"id": [0, 2, None, 3]}),
        stypes={"id": "id"},
        device=device,
    )

    left_index, right_index = join_index(
        left_table=left_table,
        right_table=right_table,
        left_keys=["id"],
        right_keys=["id"],
    )

    assert left_index.equal(torch.tensor([0, 2], device=device))
    assert right_index.equal(torch.tensor([0, 3], device=device))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_invalid_dtype(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(8, device=device),)),
    )

    with pytest.raises(TypeError, match="requires an integer input type"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=dtype,
        )


@withCUDA
def test_overflow(device: torch.device) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(200, device=device),)),
    )

    with pytest.raises(ValueError, match="row indices up to 199"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=torch.int8,
        )
