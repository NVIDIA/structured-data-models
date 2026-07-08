import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor


@pytest.fixture
def table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("num0", "num1"),
            "categorical": ("cat0", "cat1"),
        },
        numerical=torch.randn(8, 2),
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (8, 2)),
            categories=(
                StringTensor.from_list(["a", "b"]),
                torch.tensor([False, True]),
            ),
        ),
    )
