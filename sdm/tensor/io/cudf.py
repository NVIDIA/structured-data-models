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

    Raises:
        ValueError: If ``tensor`` is not CUDA-resident.
        ImportError: If cuDF is not installed.
    """
    from sdm.tensor import StringTensor

    if not tensor.is_cuda:
        raise ValueError(
            "Expected 'tensor' in 'to_cudf' to be CUDA-resident "
            f"(got '{tensor.device}')"
        )

    if isinstance(tensor, StringTensor):
        return tensor.to_cudf()

    with torch.cuda.device(tensor.device):
        try:
            import cudf
        except ImportError as exc:
            raise ImportError(
                "Converting tensors to cuDF requires cuDF"
            ) from exc

        # cuDF requires flat contiguous input, so this copies non-contiguous
        # tensors on device. `nan_as_null=False` keeps NaN values instead of
        # converting them to cuDF nulls.
        tensor = tensor.detach().contiguous().view(-1)
        return cudf.Series(tensor, copy=False, nan_as_null=False)
