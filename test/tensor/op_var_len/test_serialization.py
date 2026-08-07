import copy
import io
import pickle
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch

from sdm import VarLenTensor


@pytest.mark.parametrize(
    "round_trip",
    [
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(
            lambda tensor: pickle.loads(pickle.dumps(tensor)),
            id="pickle",
        ),
    ],
)
def test_python_serialization_round_trip(
    round_trip: Callable[[VarLenTensor], Any],
) -> None:
    tensor = VarLenTensor.from_list([[1], None, [2, 3]])
    out = round_trip(tensor)

    assert isinstance(out, VarLenTensor)
    assert out.tolist() == tensor.tolist()
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()
    assert out._valid is not None
    assert tensor._valid is not None
    assert out._valid.data_ptr() != tensor._valid.data_ptr()


def test_deepcopy_preserves_noncontiguous_layout() -> None:
    base = VarLenTensor.from_list([[[1.0], None], [[2.0], [3.0, 4.0]]])
    tensor = cast(VarLenTensor, base.t()[:, 1:])

    out = copy.deepcopy(tensor)

    assert isinstance(out, VarLenTensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.tolist() == tensor.tolist()
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()
    assert out._valid is not None
    assert tensor._valid is not None
    assert out._valid.data_ptr() != tensor._valid.data_ptr()


def test_torch_serialization_round_trip() -> None:
    tensor = VarLenTensor.from_list([[1], None, [2, 3]])
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)

    with torch.serialization.safe_globals([VarLenTensor]):
        out = torch.load(buffer, weights_only=True)

    assert isinstance(out, VarLenTensor)
    assert out.tolist() == tensor.tolist()
