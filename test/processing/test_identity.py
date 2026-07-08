import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import Identity


def test_identity_returns_input_tensor_unchanged() -> None:
    input = torch.tensor([[1.0, 2.0]])
    table = TableTensor.from_tensor(input)
    processor = Identity()

    assert processor.transform(table) is table
    assert processor.inverse_transform(table) is table


def test_identity_accepts_non_numerical_stypes() -> None:
    table = TableTensor(
        columns={
            "numerical": ("x0",),
            "categorical": ("kind",),
        },
        numerical=torch.tensor([[1.0], [2.0]]),
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )
    processor = Identity()

    assert processor.transform(table) is table
    assert processor(table) is table
