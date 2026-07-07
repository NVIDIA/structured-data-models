import torch
from sdm import TableTensor
from sdm.processing import Identity


def test_identity_returns_input_tensor_unchanged() -> None:
    input = torch.tensor([[1.0, 2.0]])
    table = TableTensor.from_tensor(input)
    processor = Identity()

    assert processor.transform(table) is table
    assert processor.inverse_transform(table) is table
