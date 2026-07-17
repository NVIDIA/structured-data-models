from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import cudf


def to_cudf(
    tensor: Tensor,
    valid_mask: Tensor | None = None,
) -> cudf.Series:
    r"""Convert a CUDA tensor to a flat :class:`cudf.Series`.

    Args:
        tensor: The CUDA tensor.
        valid_mask: Boolean mask indicating valid, non-null tensor elements.
    """
    import cudf

    from sdm.tensor import StringTensor

    if not tensor.is_cuda:
        raise ValueError(
            f"Expected tensor to be on a CUDA device (got '{tensor.device}')"
        )

    if isinstance(tensor, StringTensor):
        ser = tensor.to_cudf()
    else:
        tensor = tensor.detach().contiguous().view(-1)
        with torch.cuda.device(tensor.device):
            ser = cudf.Series(tensor, copy=False)

    if valid_mask is None:
        return ser

    if valid_mask.device != tensor.device:
        raise ValueError(
            f"Expected 'valid_mask' to be on device '{tensor.device}' "
            f"(got '{valid_mask.device}')"
        )

    column_mask = to_cudf(valid_mask.contiguous().view(-1))._column.as_mask()
    with torch.cuda.device(tensor.device):
        try:
            column = ser._column.set_mask(column_mask)
        except TypeError:
            null_count = valid_mask.numel() - int(valid_mask.sum())
            column = ser._column.set_mask(column_mask, null_count)

        return cudf.Series._from_column(column)
