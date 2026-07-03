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


def to_arrow(tensor: Tensor) -> pa.Array:
    r"""Convert a tensor to flat ``pyarrow`` list array.

    Args:
        tensor: The tensor.
    """
    tensor = tensor.detach().contiguous().view(-1).cpu()

    type = TORCH_ARROW_DTYPES.get(tensor.dtype)
    if type is None:
        raise TypeError(f"Unsupported data type '{tensor.dtype}'")

    return pa.Array.from_buffers(
        type=type,
        length=tensor.numel(),
        buffers=[None, pa.py_buffer(tensor.numpy())],
    )
