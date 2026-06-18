import torch
from schemafm import StringTensor


def test_init() -> None:
    tensor = StringTensor.from_list([["hi", "é"], ["", "abc"]])
    assert tensor.size() == (2, 2)
    assert tensor.stride() == (2, 1)
    assert tensor.dtype == torch.uint8
    assert tensor.bytes.equal(torch.tensor([104, 105, 195, 169, 97, 98, 99]))
    assert tensor.offset.equal(torch.tensor([0, 2, 4, 4, 7]))
