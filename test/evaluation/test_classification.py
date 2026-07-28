import pytest
import sdm
import torch


@pytest.mark.parametrize(
    ("columns", "categories"),
    [
        (("0", "1"), torch.tensor([1, 0])),
        (("False", "True"), torch.tensor([True, False])),
        (("A", "B"), sdm.StringTensor.from_list(["B", "A"])),
    ],
)
def test_to_class_indices(
    columns: tuple[str, ...],
    categories: torch.Tensor,
) -> None:
    x = sdm.TableTensor(
        columns={sdm.Stype.numerical: columns},
        numerical=torch.randn(4, 2),
    )
    y = sdm.CategoricalTensor(
        code=torch.tensor([[0], [1], [0], [1]]),
        categories=(categories,),
    )

    pred, target = sdm.evaluation.to_class_indices(x, y)
    assert pred.equal(x.numerical[..., [1, 0]])
    assert target.equal(torch.tensor([0, 1, 0, 1]))


@pytest.mark.parametrize(
    ("columns", "categories", "positive_class"),
    [
        (("0", "1"), torch.tensor([1, 0]), 1),
        (("False", "True"), torch.tensor([True, False]), True),
        (("A", "B"), sdm.StringTensor.from_list(["B", "A"]), "B"),
    ],
)
def test_to_binary_class(
    columns: tuple[str, ...],
    categories: torch.Tensor,
    positive_class: bool | int,
) -> None:
    x = sdm.TableTensor(
        columns={sdm.Stype.numerical: columns},
        numerical=torch.randn(4, 2),
    )
    y = sdm.CategoricalTensor(
        code=torch.tensor([[0], [1], [0], [1]]),
        categories=(categories,),
    )

    pred, target = sdm.evaluation.to_binary_class(x, y, positive_class)
    assert pred.equal(x.numerical[..., 1])
    assert target.equal(torch.tensor([True, False, True, False]))
