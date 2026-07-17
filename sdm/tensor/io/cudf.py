from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import cudf


def to_cudf(tensor: Tensor) -> cudf.Series:
    r"""Convert a CUDA tensor to a flat :class:`cudf.Series`.

    Args:
        tensor: The CUDA tensor.
    """
    from sdm.tensor import StringTensor

    if isinstance(tensor, StringTensor):
        return tensor.to_cudf()

    if not tensor.is_cuda:
        raise ValueError(
            f"Expected tensor to be on a CUDA device (got '{tensor.device}')"
        )

    tensor = tensor.detach().contiguous().view(-1)

    with torch.cuda.device(tensor.device):
        import cudf

        return cudf.Series(tensor, copy=False)
