import torch
from sdm import TableTensor
from sdm.processing import ClassShuffle, Recipe, Sequential


def _labels(values: torch.Tensor) -> TableTensor:
    return TableTensor.from_tensor(values)


def test_class_shuffle_shift_maps_labels() -> None:
    labels = _labels(torch.tensor([[0], [1], [2], [-1]]))
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes

    output = ClassShuffle(method="shift").fit_transform(labels)

    assert torch.equal(output.numerical, torch.tensor([[2], [0], [1], [-1]]))


def test_class_shuffle_random_inverse_round_trips() -> None:
    labels = _labels(torch.tensor([[0], [1], [2], [3]]))
    processor = ClassShuffle(method="random")

    transformed = processor.fit_transform(labels)
    restored = processor.inverse_transform(transformed)

    assert torch.equal(restored.numerical, labels.numerical)


def test_class_shuffle_correct_output_uses_forward_permutation() -> None:
    labels = _labels(torch.tensor([[0], [1], [2]]))
    scores = torch.tensor([[0.1, 0.2, 0.7]])
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes
    processor = ClassShuffle(method="shift").fit(labels)

    corrected = processor.correct_output(scores)

    assert torch.equal(corrected, scores[:, [2, 0, 1]])


def test_class_shuffle_learns_classes_in_recipe_target_pipeline() -> None:
    target = TableTensor.from_tensor(torch.tensor([[0], [1], [2]]))
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes
    recipe = Recipe(target=[ClassShuffle(method="shift")])

    assert isinstance(recipe.target, Sequential)
    transformed = recipe.target.fit_transform(target)
    restored = recipe.target.inverse_transform(transformed)

    assert isinstance(transformed, TableTensor)
    assert isinstance(restored, TableTensor)
    assert torch.equal(transformed.numerical, torch.tensor([[2], [0], [1]]))
    assert torch.equal(restored.numerical, target.numerical)
