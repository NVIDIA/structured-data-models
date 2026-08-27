import torch

from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import OneHot


def test_one_hot_maps_missing_and_unknown_codes_to_zero() -> None:
    table = TableTensor(
        numerical=torch.arange(4, dtype=torch.float32).unsqueeze(-1),
        categorical=CategoricalTensor(
            code=torch.tensor([[-1], [0], [1], [2]]),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )

    output = OneHot().transform(table)

    assert output.columns[Stype.numerical] == (
        "num_0",
        "cat_0__0",
        "cat_0__1",
    )
    assert output.numerical[:, 1:].tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]


def test_one_hot_uses_sum_of_uneven_cardinalities() -> None:
    table = TableTensor(
        numerical=torch.tensor([[10.0], [20.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[-1, 0, 4], [0, 1, 0]]),
            categories=(
                StringTensor.from_list([]),
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["v", "w", "x", "y", "z"]),
            ),
        ),
    )

    output = OneHot().transform(table)

    assert output.numerical.size(-1) == 1 + 2 + 5
    assert output.numerical.tolist() == [
        [10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [20.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
