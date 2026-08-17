import io

import torch

from sdm import TableTensor


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48.0).view(2, 3, 4, 2),
    )


def test_serialization_preserves_layout() -> None:
    inp = make_table()[1:, ::2]
    buffer = io.BytesIO()

    torch.save(inp, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)

    assert type(out) is TableTensor
    assert out.size() == inp.size()
    assert out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert out.columns == inp.columns
    assert out.equal(inp)
