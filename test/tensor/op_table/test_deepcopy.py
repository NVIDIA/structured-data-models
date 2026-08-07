import copy

import torch

from sdm import CategoricalTensor, TableTensor


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48.0).view(2, 3, 4, 2),
    )


def test_deepcopy_preserves_layout() -> None:
    inp = make_table()[1:, ::2]

    out = copy.deepcopy(inp)

    assert type(out) is TableTensor
    assert out.size() == inp.size()
    assert out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert out.columns == inp.columns
    assert out.equal(inp)
    assert out.numerical.data_ptr() != inp.numerical.data_ptr()


def test_deepcopy_preserves_categorical_view() -> None:
    inp = TableTensor(
        columns={"categorical": ("category",)},
        categorical=CategoricalTensor(
            code=torch.arange(6, dtype=torch.int32).view(2, 3, 1),
            categories=(torch.arange(6),),
        ),
    )[1:]

    out = copy.deepcopy(inp)

    assert type(out) is TableTensor
    assert out.size() == inp.size()
    assert out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert out.columns == inp.columns
    assert out.categorical.equal(inp.categorical)
    assert out.categorical.stride() == inp.categorical.stride()
    assert out.categorical.storage_offset() == inp.categorical.storage_offset()
    assert (
        out.categorical.code.untyped_storage().data_ptr()
        != inp.categorical.code.untyped_storage().data_ptr()
    )
