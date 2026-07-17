from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any

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


def _set_cudf_mask(column: Any, mask: Any, null_count: int) -> Any:
    r"""Set a cuDF column mask across cuDF versions."""
    try:
        parameters = signature(column.set_mask).parameters
    except (TypeError, ValueError):
        try:
            return column.set_mask(mask, null_count)
        except TypeError as exc:
            try:
                return column.set_mask(mask)
            except TypeError:
                raise exc from None

    if "null_count" in parameters or len(parameters) > 1:
        return column.set_mask(mask, null_count)

    return column.set_mask(mask)
