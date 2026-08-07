from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from sdm import VarLenTensor


@pytest.mark.parametrize(
    ("operation", "op_name"),
    [
        pytest.param(lambda tensor: tensor + 1, "aten.add.Tensor", id="add"),
        pytest.param(
            lambda tensor: tensor.sum(), "aten.sum.default", id="sum"
        ),
        pytest.param(
            lambda tensor: tensor.sin(), "aten.sin.default", id="sin"
        ),
        pytest.param(
            lambda tensor: tensor.copy_(tensor.clone()),
            "aten.copy_.default",
            id="copy_",
        ),
        pytest.param(
            lambda tensor: torch.index_select(
                tensor,
                dim=0,
                index=torch.tensor([0]),
                out=cast(Tensor, tensor),
            ),
            "aten.index_select.out",
            id="out",
        ),
    ],
)
def test_unsupported_semantics_fail_explicitly(
    operation: Callable[[VarLenTensor], Any],
    op_name: str,
) -> None:
    tensor = VarLenTensor.from_list([[1], [2, 3]])

    with pytest.raises(NotImplementedError, match=rf"'{op_name}'"):
        operation(tensor)
