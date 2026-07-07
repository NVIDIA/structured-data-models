import pytest
import torch
from sdm import TableTensor
from sdm.processing import LabelShuffle, Recipe, Sequential


def test_label_shuffle_default_single_estimator_is_identity() -> None:
    labels = torch.tensor([[0], [1], [2]])

    output = LabelShuffle().fit_transform(labels)

    assert output is labels


def test_label_shuffle_resolved_shift_maps_labels() -> None:
    labels = torch.tensor([[0], [1], [2], [-1]])
    processor = LabelShuffle(n_classes=3).resolve(estimator=1)

    output = processor.transform(labels)

    assert torch.equal(output, torch.tensor([[2], [0], [1], [-1]]))


def test_label_shuffle_inverse_restores_shifted_labels() -> None:
    labels = torch.tensor([[0], [1], [2]])
    processor = LabelShuffle(n_classes=3).resolve(estimator=1)

    transformed = processor.transform(labels)
    restored = processor.inverse_transform(transformed)

    assert torch.equal(restored, labels)


def test_label_shuffle_correct_output_uses_forward_permutation() -> None:
    scores = torch.tensor([[0.1, 0.2, 0.7]])
    processor = LabelShuffle(n_classes=3).resolve(estimator=1)

    corrected = processor.correct_output(scores)

    assert torch.equal(corrected, scores[:, [2, 0, 1]])


def test_label_shuffle_learns_classes_in_recipe_target_pipeline() -> None:
    target = TableTensor.from_tensor(torch.tensor([[0], [1], [2]]))
    recipe = Recipe(target=[LabelShuffle().resolve(estimator=1)])

    assert isinstance(recipe.target, Sequential)
    transformed = recipe.target.fit_transform(target)
    restored = recipe.target.inverse_transform(transformed)

    assert isinstance(transformed, TableTensor)
    assert isinstance(restored, TableTensor)
    assert torch.equal(transformed.numerical, torch.tensor([[2], [0], [1]]))
    assert torch.equal(restored.numerical, target.numerical)


def test_label_shuffle_random_is_deterministic() -> None:
    labels = torch.tensor([[0], [1], [2], [3]])
    generator = torch.Generator().manual_seed(123)
    processor = LabelShuffle(
        method="random",
        n_classes=4,
        generator=generator,
    ).resolve(estimator=2)

    first = processor.transform(labels)
    second = processor.transform(labels)

    assert torch.equal(first, second)


def test_label_shuffle_rejects_fractional_labels() -> None:
    processor = LabelShuffle()

    with pytest.raises(ValueError, match="integer-valued"):
        processor.fit(torch.tensor([[0.0], [1.5]]))


def test_label_shuffle_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        LabelShuffle(method="bad")  # ty: ignore[invalid-argument-type]


def test_label_shuffle_identity_still_validates_label_bounds() -> None:
    processor = LabelShuffle(n_classes=3)

    with pytest.raises(ValueError, match="less than n_classes"):
        processor.transform(torch.tensor([[4]]))


def test_label_shuffle_identity_still_validates_output_shape() -> None:
    processor = LabelShuffle(n_classes=3)

    with pytest.raises(ValueError, match="output class dimension"):
        processor.correct_output(torch.ones(2, 2))


def test_label_shuffle_rejects_output_shape_mismatch() -> None:
    processor = LabelShuffle(n_classes=3).resolve(estimator=1)

    with pytest.raises(ValueError, match="output class dimension"):
        processor.correct_output(torch.ones(2, 2))
