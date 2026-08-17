from typing import cast

import torch
from torch import Tensor

from sdm import TableTensor

aten = torch.ops.aten


def make_table(*, requires_grad: bool = False) -> TableTensor:
    numerical = torch.arange(48.0).view(2, 3, 4, 2)
    numerical.requires_grad_(requires_grad)
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=numerical,
    )


def assert_matches_dense(out: TableTensor, expected: Tensor) -> None:
    assert type(out) is TableTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.layout == expected.layout
    assert out.numerical.equal(expected)


def test_alias_and_detach_preserve_aliases() -> None:
    inp = cast(
        TableTensor,
        make_table(requires_grad=True).transpose(0, 1),
    )

    alias = aten.alias.default(inp)
    detached = aten.detach.default(inp)

    assert_matches_dense(alias, inp.numerical)
    assert_matches_dense(detached, inp.numerical)
    assert alias.numerical.data_ptr() == inp.numerical.data_ptr()
    assert detached.numerical.data_ptr() == inp.numerical.data_ptr()
    assert not detached.numerical.requires_grad


def test_alias_and_detach_preserve_input_inference_state() -> None:
    normal = make_table()
    with torch.inference_mode():
        normal_outputs = (
            aten.alias.default(normal),
            aten.detach.default(normal),
        )
        inference = make_table()

    inference_outputs = (
        aten.alias.default(inference),
        aten.detach.default(inference),
    )

    assert all(not out.is_inference() for out in normal_outputs)
    assert all(out.is_inference() for out in inference_outputs)
