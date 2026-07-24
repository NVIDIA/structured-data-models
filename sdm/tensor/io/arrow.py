import warnings

import pyarrow as pa
import torch
from torch import Tensor

ARROW_TORCH_DTYPES = {
    # TODO Support boolean dtype.
    # TODO Support bfloat16 dtype.
    pa.uint8(): torch.uint8,
    pa.uint16(): torch.uint16,
    pa.uint32(): torch.uint32,
    pa.uint64(): torch.uint64,
    pa.int8(): torch.int8,
    pa.int16(): torch.int16,
    pa.int32(): torch.int32,
    pa.int64(): torch.int64,
    pa.float16(): torch.float16,
    pa.float32(): torch.float32,
    pa.float64(): torch.float64,
}
TORCH_ARROW_DTYPES = {value: key for key, value in ARROW_TORCH_DTYPES.items()}


def arrow_as_tensor(
    array: pa.Array | pa.ChunkedArray,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Convert a :class:`pyarrow.Array` to a tensor.

    Args:
        array: The :class:`pyarrow.Array` or :class:`pyarrow.ChunkedArray`.
        dtype: The dtype.
        device: The device.
    """
    values = array.to_numpy(zero_copy_only=False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
        )
        return torch.as_tensor(values, dtype=dtype, device=device)


def to_arrow(tensor: Tensor) -> pa.Array:
    r"""Convert a tensor to a flat :class:`pyarrow.Array`.

    Args:
        tensor: The tensor.
    """
    # Avoid a circular import through `sdm.tensor`.
    from sdm.tensor import VarLenTensor  # noqa: PLC0415

    if isinstance(tensor, VarLenTensor):
        return tensor.to_arrow()

    tensor = tensor.detach().contiguous().view(-1).cpu()

    arrow_type = TORCH_ARROW_DTYPES.get(tensor.dtype)
    if arrow_type is None:
        raise TypeError(f"Unsupported data type '{tensor.dtype}'")

    return pa.Array.from_buffers(
        type=arrow_type,
        length=tensor.numel(),
        buffers=[None, pa.py_buffer(tensor.numpy())],
    )
