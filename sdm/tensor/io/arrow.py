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


def to_arrow(tensor: Tensor, valid: Tensor | None = None) -> pa.Array:
    r"""Convert a tensor to a flat :class:`pyarrow.Array`.

    Args:
        tensor: The tensor.
        valid: The validity bitmap.
    """
    from sdm.tensor import VarLenTensor  # noqa: PLC0415

    if isinstance(tensor, VarLenTensor):
        array = tensor.to_arrow()
        if valid is not None:
            valid = valid.contiguous().view(-1).cpu()
            if array.offset > 0:
                valid = torch.cat([valid.new_ones(array.offset), valid])
            buffers = list(array.buffers()[: array.type.num_buffers])
            buffers[0] = pa.array(valid.numpy(), type=pa.bool_()).buffers()[1]
            array = pa.Array.from_buffers(
                type=array.type,
                length=len(array),
                buffers=buffers,
                null_count=-1,
                offset=array.offset,
                children=[array.values]
                if pa.types.is_list(array.type)
                or pa.types.is_large_list(array.type)
                else None,
            )
        return array

    tensor = tensor.detach().contiguous().view(-1).cpu()
    if valid is not None:
        valid = valid.contiguous().view(-1).cpu()

    arrow_type = TORCH_ARROW_DTYPES.get(tensor.dtype)
    if arrow_type is None:
        raise TypeError(f"Unsupported data type '{tensor.dtype}'")

    return pa.Array.from_buffers(
        type=arrow_type,
        length=tensor.numel(),
        buffers=[
            pa.array(valid.numpy(), type=pa.bool_()).buffers()[1]
            if valid is not None
            else None,
            pa.py_buffer(tensor.numpy()),
        ],
    )
