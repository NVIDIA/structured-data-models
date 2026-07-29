from typing import Any, cast

import pyarrow as pa
import pytest
import torch

from sdm import MultiCategoricalTensor, StringTensor, VarLenTensor
from sdm.testing import onlyCUDA


@pytest.mark.parametrize(
    "list_type",
    [pa.list_(pa.string()), pa.large_list(pa.string())],
)
def test_arrow(list_type: pa.DataType) -> None:
    array = pa.array(
        [["red", "blue"], None, [], ["blue", None]],
        type=list_type,
    )
    tensor = MultiCategoricalTensor.from_arrow(array)

    assert tensor.dtype == torch.int32
    assert tensor.size() == (4, 1)
    assert tensor.code.tolist() == [[[0, 1]], [[-2]], [[]], [[1, -1]]]
    assert tensor.categories[0].tolist() == ["red", "blue"]
    assert tensor.tolist() == [
        [["red", "blue"]],
        [None],
        [[]],
        [["blue", None]],
    ]
    assert tensor.isfinite().equal(
        torch.tensor([[True], [False], [True], [True]])
    )
    assert tensor.isnan().equal(
        torch.tensor([[False], [True], [False], [False]])
    )

    out = tensor.to_arrow(names=["tags"])
    assert out.column_names == ["tags"]
    assert (
        pa.types.is_list(out["tags"].type)
        if pa.types.is_list(list_type)
        else pa.types.is_large_list(out["tags"].type)
    )
    assert out.to_pydict() == {"tags": array.to_pylist()}


def test_arrow_chunked_numeric() -> None:
    tensor = MultiCategoricalTensor.from_arrow(
        pa.chunked_array(
            [
                pa.array([[10, 20], None]),
                pa.array([[], [20, 10]]),
            ]
        ),
        dtype=torch.int64,
    )

    assert tensor.dtype == torch.int64
    assert tensor.categories[0].equal(torch.tensor([10, 20]))
    assert tensor.to_arrow().to_pydict() == {
        "0": [[10, 20], None, [], [20, 10]]
    }


def test_init_requires_variable_length_codes() -> None:
    with pytest.raises(TypeError, match="VarLenTensor"):
        MultiCategoricalTensor(
            code=cast(VarLenTensor, torch.tensor([[0, 1]])),
            categories=(torch.arange(2), torch.arange(2)),
        )


def test_null_code_contract() -> None:
    tensor = MultiCategoricalTensor(
        code=VarLenTensor.from_list([[[-2]], [[-1]], [[]]]),
        categories=(torch.empty(0, dtype=torch.int64),),
    )
    assert tensor.tolist() == [[None], [[None]], [[]]]

    invalid = MultiCategoricalTensor(
        code=VarLenTensor.from_list([[[-2, -1]]]),
        categories=(torch.empty(0, dtype=torch.int64),),
    )
    with pytest.raises(ValueError, match="singleton '-2'"):
        invalid.to_arrow()

    with pytest.raises(NotImplementedError, match="dense tensor"):
        MultiCategoricalTensor.from_tensor(torch.tensor([[0]]))


def test_tensor_ops() -> None:
    tensor = MultiCategoricalTensor.from_arrow(
        pa.array([["a", "b"], [], ["b"]]),
    )

    sliced = tensor[[2, 0]]
    assert isinstance(sliced, MultiCategoricalTensor)
    assert sliced.tolist() == [[["b"]], [["a", "b"]]]

    cloned = tensor.clone()
    assert isinstance(cloned, MultiCategoricalTensor)
    assert cloned.equal(tensor)

    concatenated = torch.cat([tensor, tensor], dim=0)
    assert isinstance(concatenated, MultiCategoricalTensor)
    assert concatenated.size() == (6, 1)
    assert concatenated.tolist() == tensor.tolist() * 2

    wide = torch.cat([tensor, tensor], dim=-1)
    assert isinstance(wide, MultiCategoricalTensor)
    assert wide.size() == (3, 2)
    assert wide.to_arrow(names=["left", "right"]).to_pydict() == {
        "left": [["a", "b"], [], ["b"]],
        "right": [["a", "b"], [], ["b"]],
    }

    stacked = torch.stack([tensor, tensor], dim=0)
    assert isinstance(stacked, MultiCategoricalTensor)
    assert stacked.size() == (2, 3, 1)

    code = cast(VarLenTensor, tensor.code.to(torch.int64))
    converted = MultiCategoricalTensor(code, tensor.categories)
    assert converted.dtype == torch.int64


def test_string_categories() -> None:
    tensor = MultiCategoricalTensor.from_arrow(
        pa.array([["a"], ["b", "a"]]),
    )
    assert isinstance(tensor.categories[0], StringTensor)


def test_cudf_io_is_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="from cuDF"):
        MultiCategoricalTensor.from_cudf(cast(Any, None))

    tensor = MultiCategoricalTensor.from_arrow(pa.array([["a"]]))
    with pytest.raises(NotImplementedError, match="to cuDF"):
        tensor.to_cudf()


@onlyCUDA
def test_to_device_cuda() -> None:
    tensor = MultiCategoricalTensor.from_arrow(
        pa.array([["a", "b"], None, []]),
        device="cuda",
    )

    assert tensor.is_cuda
    assert tensor.code.is_cuda
    assert tensor.categories[0].is_cuda
    assert tensor.to_arrow().to_pydict() == {"0": [["a", "b"], None, []]}
