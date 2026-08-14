import warnings

import pyarrow as pa
import torch
from torch import Tensor

ARROW_TORCH_DTYPES = {
    # TODO Support bfloat16 dtype.
    pa.bool_(): torch.bool,
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


def _combine_arrow_chunks(array: pa.ChunkedArray) -> pa.Array:
    r"""Combine chunks after promoting 32-bit string offsets."""
    if array.num_chunks == 0:
        return array.combine_chunks()
    if array.num_chunks == 1:
        return array.chunk(0)

    if pa.types.is_string(array.type):
        array = array.cast(pa.large_string())
    elif pa.types.is_dictionary(array.type) and pa.types.is_string(
        array.type.value_type
    ):
        array = array.cast(
            pa.dictionary(
                array.type.index_type,
                pa.large_string(),
                ordered=array.type.ordered,
            )
        )

    return array.combine_chunks()


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


def to_arrow(tensor: Tensor, valid_mask: Tensor | None = None) -> pa.Array:
    r"""Convert a tensor to a flat :class:`pyarrow.Array`.

    Args:
        tensor: The tensor.
        valid_mask: Boolean mask indicating valid, non-null tensor elements.
    """
    tensor = tensor.detach().contiguous().view(-1).cpu()
    if valid_mask is not None:
        valid_mask = valid_mask.contiguous().view(-1).cpu()

    if tensor.dtype == torch.bool:
        array = pa.array(tensor.numpy(), type=pa.bool_())
        if valid_mask is None:
            return array

        return pa.Array.from_buffers(
            type=pa.bool_(),
            length=tensor.numel(),
            buffers=[
                pa.array(valid_mask.numpy(), type=pa.bool_()).buffers()[1],
                array.buffers()[1],
            ],
            null_count=-1,
        )

    arrow_type = TORCH_ARROW_DTYPES.get(tensor.dtype)
    if arrow_type is None:
        raise TypeError(f"Unsupported data type '{tensor.dtype}'")

    return pa.Array.from_buffers(
        type=arrow_type,
        length=tensor.numel(),
        buffers=[
            pa.array(valid_mask.numpy(), type=pa.bool_()).buffers()[1]
            if valid_mask is not None
            else None,
            pa.py_buffer(tensor.numpy()),
        ],
        null_count=-1 if valid_mask is not None else 0,
    )
