import copy
import io

import torch

from sdm import ColumnarTensor


def make_columnar() -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        ),
    )


def test_deepcopy_and_serialization_preserve_layout() -> None:
    inp = make_columnar()[1:, ::2]

    deep = copy.deepcopy(inp)
    buffer = io.BytesIO()
    torch.save(inp, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    for out in (deep, loaded):
        assert type(out) is ColumnarTensor
        assert out.size() == inp.size()
        assert out.stride() == inp.stride()
        assert out.storage_offset() == inp.storage_offset()
        assert out.tolist() == inp.tolist()
    for actual, source in zip(deep.unbind(-1), inp.unbind(-1)):
        assert actual.data_ptr() != source.data_ptr()
