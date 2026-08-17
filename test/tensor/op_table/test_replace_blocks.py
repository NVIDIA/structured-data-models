from typing import cast

import torch

from sdm import TableTensor


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48).view(2, 3, 4, 2),
    )


def test_replace_blocks_preserves_logical_layout() -> None:
    inp = cast(TableTensor, make_table().transpose(0, 1)[1:])
    numerical = inp.numerical.clone(memory_format=torch.preserve_format)

    out = inp.replace_blocks(numerical=numerical)

    assert type(out) is TableTensor
    assert out.size() == inp.size()
    assert out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert out.columns == inp.columns
    assert out.numerical is numerical
