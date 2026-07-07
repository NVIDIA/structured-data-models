import torch
from sdm import TableTensor
from sdm.processing import Identity


def test_identity_returns_input_tensor_unchanged() -> None:
    input = torch.tensor([[1.0, 2.0]])
    table = TableTensor.from_tensor(input)
    processor = Identity()

    assert processor.transform(table) is not table
    assert torch.equal(processor.transform(table).numerical, input)
    assert torch.equal(processor.inverse_transform(table).numerical, input)
