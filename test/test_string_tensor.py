import pytest
import torch
from schemafm import StringTensor


def test_init() -> None:
    strings = StringTensor([["hi", "é"], ["", "abc"]])

    assert strings.size() == (2, 2)
    assert strings.stride() == (2, 1)
    assert strings.dtype == torch.uint8
    assert strings.tolist() == [["hi", "é"], ["", "abc"]]
    assert strings.offsets.equal(torch.tensor([0, 2, 4, 4, 7]))
    assert strings.as_bytes().equal(
        torch.tensor([104, 105, 195, 169, 97, 98, 99], dtype=torch.uint8)
    )


def test_shape_views() -> None:
    strings = StringTensor([["a", "bb"], ["ccc", "dddd"]])

    assert strings[1].tolist() == ["ccc", "dddd"]
    assert strings[:, 1].tolist() == ["bb", "dddd"]
    assert strings[:, ::2].tolist() == [["a"], ["ccc"]]


def test_copy_and_blocked_ops() -> None:
    strings = StringTensor(["a", "bb"])

    assert strings.clone().tolist() == ["a", "bb"]
    assert strings.detach().tolist() == ["a", "bb"]
    assert strings.to("cpu").tolist() == ["a", "bb"]

    with pytest.raises(NotImplementedError):
        strings + strings
