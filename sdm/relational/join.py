import importlib.util
import math
import warnings
from collections.abc import Sequence
from typing import Literal

import pyarrow as pa
import torch
from torch import Tensor

from sdm import TableTensor
from sdm.tensor.io import arrow_as_tensor, to_cudf

PREFIX = "sdm_internal"
LEFT_ROW_ID = f"__{PREFIX}_left_row_id__"
RIGHT_ROW_ID = f"__{PREFIX}_right_row_id__"

_warned = False


def join_index(
    left_table: TableTensor,
    right_table: TableTensor,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    how: Literal["inner"] = "inner",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Return row-index pairs for matching rows in two tables.

    Args:
        left_table: The left table.
        right_table: The right table.
        left_keys: Column names from ``left_table`` used as join keys.
        right_keys: Column names from ``right_table`` used as join keys.
        how: The join type.
        dtype: The dtype.
        device: The device.

    Returns:
        ``(left_index, right_index)`` pair with one entry per matched row.
    """
    global _warned
    assert how == "inner"

    if left_table.device != right_table.device:
        raise RuntimeError(
            "Expected 'left_table' and 'right_table' to be on the same device "
            f"(got '{left_table.device}' and '{right_table.device}')"
        )

    device = device or left_table.device

    left_table = left_table[list(left_keys)]
    right_table = right_table[list(right_keys)]
    left_rows = math.prod(left_table.size()[:-1])
    right_rows = math.prod(right_table.size()[:-1])

    backend: Literal["arrow", "cudf"] = "arrow"
    if left_table.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        elif torch.is_warn_always_enabled() or not _warned:
            message = (
                "Falling back to a CPU-based join because cuDF is not "
                "installed. Install cuDF to enable faster CUDA-based joins "
                "without device synchronization."
            )
            if not torch.is_warn_always_enabled():
                message += (
                    " This warning will be suppressed for the remainder of "
                    "this process."
                )

            warnings.warn(message, RuntimeWarning, stacklevel=2)
            _warned = True

    if backend == "arrow":
        left = left_table.to_arrow().append_column(
            LEFT_ROW_ID,
            pa.array(torch.arange(left_rows, dtype=dtype).numpy()),
        )
        right = right_table.to_arrow().append_column(
            RIGHT_ROW_ID,
            pa.array(torch.arange(right_rows, dtype=dtype).numpy()),
        )

        joined = left.join(
            right,
            keys=left_keys,
            right_keys=right_keys,
            join_type=how,
        )

        return (
            arrow_as_tensor(joined[LEFT_ROW_ID], device=device),
            arrow_as_tensor(joined[RIGHT_ROW_ID], device=device),
        )

    assert backend == "cudf"
    with torch.cuda.device(left_table.device):
        left = left_table.to_cudf()
        left[LEFT_ROW_ID] = to_cudf(
            torch.arange(left_rows, dtype=dtype, device=left_table.device)
        )
        right = right_table.to_cudf()
        right[RIGHT_ROW_ID] = to_cudf(
            torch.arange(right_rows, dtype=dtype, device=right_table.device)
        )

        joined = left.merge(
            right,
            left_on=left_keys,
            right_on=right_keys,
            how=how,
        )

        return (
            torch.as_tensor(joined[LEFT_ROW_ID]).to(device),
            torch.as_tensor(joined[RIGHT_ROW_ID]).to(device),
        )
