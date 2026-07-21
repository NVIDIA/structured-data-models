import pandas as pd
import pytest
import torch
from sdm import TableTensor, infer_stypes
from sdm.relational.join import join_index
from sdm.testing import withCUDA


def table(num_rows: int, device: torch.device) -> TableTensor:
    df = pd.DataFrame({"id": range(num_rows)})
    return TableTensor.from_pandas(
        df=df,
        stypes=infer_stypes(df),
        device=device,
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int8, torch.int32])
def test_join_index_dtype(dtype: torch.dtype, device: torch.device) -> None:
    num_rows = 8 if dtype == torch.int8 else 5000
    left_table = right_table = table(num_rows, device)

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
    # Self-join on unique keys must yield every row index exactly once:
    expected = torch.arange(num_rows, dtype=dtype, device=device)
    assert left_index.sort().values.equal(expected)
    assert right_index.equal(left_index)


@withCUDA
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.uint8],
)
def test_join_index_invalid_dtype(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    left_table = right_table = table(8, device)

    with pytest.raises(ValueError, match="signed integer"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=dtype,
        )


@withCUDA
def test_join_index_too_narrow_dtype(device: torch.device) -> None:
    left_table = right_table = table(200, device)

    with pytest.raises(ValueError, match="row indices up to 199"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=torch.int8,
        )
