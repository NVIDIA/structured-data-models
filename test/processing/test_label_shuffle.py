import pytest
import torch
from sdm import TableTensor
from sdm.processing import LabelShuffle, Recipe, Sequential


def _labels(values: torch.Tensor) -> TableTensor:
    return TableTensor.from_tensor(values)


def test_label_shuffle_none_is_identity() -> None:
    labels = _labels(torch.tensor([[0], [1], [2]]))

    output = LabelShuffle(method="none").fit_transform(labels)

    assert output is labels


def test_label_shuffle_shift_maps_labels() -> None:
    labels = _labels(torch.tensor([[0], [1], [2], [-1]]))
    torch.manual_seed(0)  # draws a cyclic offset of 1 for three classes
    processor = LabelShuffle(n_classes=3)

    output = processor.transform(labels)

    assert torch.equal(output.numerical, torch.tensor([[2], [0], [1], [-1]]))


def test_label_shuffle_inverse_restores_shifted_labels() -> None:
    labels = _labels(torch.tensor([[0], [1], [2]]))
    processor = LabelShuffle(n_classes=3)

    transformed = processor.transform(labels)
    restored = processor.inverse_transform(transformed)

    assert torch.equal(restored.numerical, labels.numerical)


def test_label_shuffle_correct_output_uses_forward_permutation() -> None:
    scores = torch.tensor([[0.1, 0.2, 0.7]])
    torch.manual_seed(0)  # draws a cyclic offset of 1 for three classes
    processor = LabelShuffle(n_classes=3)

    corrected = processor.correct_output(scores)

    assert torch.equal(corrected, scores[:, [2, 0, 1]])


def test_label_shuffle_learns_classes_in_recipe_target_pipeline() -> None:
    target = TableTensor.from_tensor(torch.tensor([[0], [1], [2]]))
    torch.manual_seed(0)  # draws a cyclic offset of 1 for three classes
    recipe = Recipe(target=[LabelShuffle()])

    assert isinstance(recipe.target, Sequential)
    transformed = recipe.target.fit_transform(target)
    restored = recipe.target.inverse_transform(transformed)

    assert isinstance(transformed, TableTensor)
    assert isinstance(restored, TableTensor)
    assert torch.equal(transformed.numerical, torch.tensor([[2], [0], [1]]))
    assert torch.equal(restored.numerical, target.numerical)


def test_label_shuffle_same_global_seed_draws_same_view() -> None:
    labels = _labels(torch.tensor([[0], [1], [2], [3]]))

    torch.manual_seed(123)
    first = LabelShuffle(method="random", n_classes=4).transform(labels)
    torch.manual_seed(123)
    second = LabelShuffle(method="random", n_classes=4).transform(labels)

    assert torch.equal(first.numerical, second.numerical)


def test_label_shuffle_transform_is_deterministic_per_instance() -> None:
    labels = _labels(torch.tensor([[0], [1], [2], [3]]))
    processor = LabelShuffle(method="random", n_classes=4)

    first = processor.transform(labels)
    second = processor.transform(labels)

    assert torch.equal(first.numerical, second.numerical)


def test_label_shuffle_rejects_fractional_labels() -> None:
    processor = LabelShuffle()

    with pytest.raises(ValueError, match="integer-valued"):
        processor.fit(_labels(torch.tensor([[0.0], [1.5]])))


def test_label_shuffle_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        LabelShuffle(method="bad")  # ty: ignore[invalid-argument-type]


def test_label_shuffle_validates_label_bounds() -> None:
    processor = LabelShuffle(n_classes=3)

    with pytest.raises(ValueError, match="less than n_classes"):
        processor.transform(_labels(torch.tensor([[4]])))


def test_label_shuffle_validates_output_shape() -> None:
    processor = LabelShuffle(n_classes=3)

    with pytest.raises(ValueError, match="output class dimension"):
        processor.correct_output(torch.ones(2, 2))
