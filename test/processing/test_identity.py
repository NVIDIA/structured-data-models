import torch
from sdm.processing import Identity


def test_identity_returns_input_tensor_unchanged() -> None:
    input = torch.tensor([[1.0, 2.0]])
    processor = Identity()

    assert processor.transform(input) is input
    assert processor.inverse_transform(input) is input
