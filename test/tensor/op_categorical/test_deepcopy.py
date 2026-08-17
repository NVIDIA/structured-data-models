import copy
import io
import pickle

import pytest
import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor, TableTensor


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _tensor() -> CategoricalTensor:
    return CategoricalTensor(
        code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        categories=(
            StringTensor.from_list(["a", "b"]),
            torch.tensor([10.0, 20.0], requires_grad=True),
        ),
    )


def _noncontiguous_tensor(*, inference: bool = False) -> CategoricalTensor:
    with torch.inference_mode(inference):
        tensor = CategoricalTensor(
            code=torch.arange(24, dtype=torch.int32)
            .remainder(4)
            .view(2, 3, 4),
            categories=tuple(torch.arange(4) + 10 * i for i in range(4)),
        )
        out = tensor[:, 1:]
    assert isinstance(out, CategoricalTensor)
    return out


def test_pickle_save_and_deepcopy_round_trip() -> None:
    tensor = _tensor()
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)

    outputs = (
        copy.deepcopy(tensor),
        pickle.loads(pickle.dumps(tensor)),
        torch.load(buffer, weights_only=False),
    )
    for out in outputs:
        assert isinstance(out, CategoricalTensor)
        assert out.tolist() == tensor.tolist()
        assert out.size() == tensor.size()
        assert out.stride() == tensor.stride()
        assert out.dtype == tensor.dtype
        assert not _is_alias_of(out.code, tensor.code)


@pytest.mark.parametrize("inference", [False, True])
def test_deepcopy_noncontiguous_view_preserves_layout(
    inference: bool,
) -> None:
    tensor = _noncontiguous_tensor(inference=inference)

    with torch.inference_mode(not inference):
        out = copy.deepcopy(tensor)

    assert isinstance(out, CategoricalTensor)
    assert out.is_inference() == tensor.is_inference()
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.tolist() == tensor.tolist()
    assert not _is_alias_of(out.code, tensor.code)
    assert all(
        not _is_alias_of(actual, expected)
        for actual, expected in zip(out.categories, tensor.categories)
    )


def test_deepcopy_table_with_noncontiguous_categorical_block() -> None:
    categorical = _noncontiguous_tensor()
    table = TableTensor(
        columns={"categorical": ("a", "b", "c", "d")},
        categorical=categorical,
    )

    out = copy.deepcopy(table)

    assert isinstance(out, TableTensor)
    assert out.columns == table.columns
    assert out.equal(table)
    assert out.categorical.stride() == categorical.stride()
    assert out.categorical.storage_offset() == categorical.storage_offset()
    assert not _is_alias_of(out.categorical.code, categorical.code)
