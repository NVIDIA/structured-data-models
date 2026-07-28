import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.evaluation import to_binary_class, to_class_indices


def _prediction(
    columns: tuple[str, ...],
) -> TableTensor:
    return TableTensor(
        columns={Stype.numerical: columns},
        numerical=torch.arange(8, dtype=torch.float).view(4, 2),
    )


@pytest.mark.parametrize(
    ("columns", "categories"),
    [
        (("False", "True"), torch.tensor([True, False])),
        (("0", "1"), torch.tensor([1, 0])),
        (("0.0", "1.0"), torch.tensor([1.0, 0.0])),
    ],
)
def test_to_class_indices(
    columns: tuple[str, ...],
    categories: torch.Tensor,
) -> None:
    pred = _prediction(columns)
    target = CategoricalTensor(
        code=torch.tensor([[0], [1], [0], [1]]),
        categories=(categories,),
    )

    score, target_index = to_class_indices(pred, target)

    assert score is pred.numerical
    assert target_index.tolist() == [1, 0, 1, 0]


def test_to_class_indices_rejects_string_categories() -> None:
    pred = _prediction(("A", "B"))
    target = CategoricalTensor(
        code=torch.tensor([[0], [1], [0], [1]]),
        categories=(StringTensor.from_list(["B", "A"]),),
    )

    with pytest.raises(NotImplementedError, match="string categories"):
        to_class_indices(pred, target)


def test_to_class_indices_accepts_table_target() -> None:
    pred = _prediction(("0", "1"))
    target = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0], [1]]),
            categories=(torch.tensor([1, 0]),),
        ),
    )

    _, target_index = to_class_indices(pred, target)

    assert target_index.tolist() == [1, 0, 1, 0]


def test_to_class_indices_rejects_missing_prediction_class() -> None:
    pred = _prediction(("0", "1"))
    target = CategoricalTensor(
        code=torch.tensor([[0], [1], [0], [1]]),
        categories=(torch.tensor([1, 2]),),
    )

    with pytest.raises(ValueError, match="missing from prediction"):
        to_class_indices(pred, target)


def test_to_class_indices_rejects_missing_target_values() -> None:
    pred = _prediction(("0", "1"))
    target = CategoricalTensor(
        code=torch.tensor([[0], [-1], [0], [-1]]),
        categories=(torch.tensor([1, 0]),),
    )

    with pytest.raises(ValueError, match="missing values"):
        to_class_indices(pred, target)


@pytest.mark.parametrize(
    ("columns", "categories", "positive_class"),
    [
        (("False", "True"), torch.tensor([True, False]), True),
        (("0", "1"), torch.tensor([1, 0]), 1),
        (("0.0", "1.0"), torch.tensor([1.0, 0.0]), 1.0),
    ],
)
def test_to_binary_class(
    columns: tuple[str, ...],
    categories: torch.Tensor,
    positive_class: bool | int | float,
) -> None:
    pred = _prediction(columns)
    target = CategoricalTensor(
        code=torch.tensor([[0], [1], [0], [1]]),
        categories=(categories,),
    )

    score, binary_target = to_binary_class(
        pred,
        target,
        positive_class=positive_class,
    )

    assert score.tolist() == [1.0, 3.0, 5.0, 7.0]
    assert binary_target.tolist() == [True, False, True, False]


def test_to_binary_class_rejects_missing_positive_prediction_class() -> None:
    pred = _prediction(("0", "1"))
    target = CategoricalTensor(
        code=torch.tensor([[0], [1], [0], [1]]),
        categories=(torch.tensor([1, 0]),),
    )

    with pytest.raises(ValueError, match="prediction classes"):
        to_binary_class(pred, target, positive_class=2)


def test_to_binary_class_rejects_missing_positive_target_class() -> None:
    pred = _prediction(("0", "1"))
    target = CategoricalTensor(
        code=torch.tensor([[0], [0], [0], [0]]),
        categories=(torch.tensor([0]),),
    )

    with pytest.raises(ValueError, match="target categories"):
        to_binary_class(pred, target, positive_class=1)
