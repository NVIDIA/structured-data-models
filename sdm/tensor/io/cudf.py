from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import cudf


def to_cudf(
    tensor: Tensor,
    valid: Tensor | None = None,
) -> cudf.Series:
    r"""Convert a CUDA tensor to a flat :class:`cudf.Series`.

    Args:
        tensor: The CUDA tensor.
        valid: The validity bitmap.
    """
    import cudf

    from sdm.tensor import StringTensor  # noqa: PLC0415

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

    if valid is None:
        return ser

    if valid.device != tensor.device:
        raise ValueError(
            f"Expected 'valid' to be on device '{tensor.device}' "
            f"(got '{valid.device}')"
        )

    with torch.cuda.device(tensor.device):
        mask = to_cudf(valid.contiguous().view(-1))._column.as_mask()
        if not isinstance(mask, tuple):
            mask = (mask,)

        return cudf.Series._from_column(ser._column.set_mask(*mask))
