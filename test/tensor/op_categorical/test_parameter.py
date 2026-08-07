import pytest
import torch

from sdm import CategoricalTensor


def test_parameter_support_requires_grad_false() -> None:
    tensor = CategoricalTensor(
        code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        categories=(torch.tensor([0, 1]), torch.tensor([10, 20])),
    )
    parameter = torch.nn.Parameter(tensor, requires_grad=False)

    assert isinstance(parameter, CategoricalTensor)
    assert isinstance(parameter.detach(), CategoricalTensor)
    assert not parameter.requires_grad

    with pytest.raises(RuntimeError, match="floating point dtype"):
        torch.nn.Parameter(tensor, requires_grad=True)
