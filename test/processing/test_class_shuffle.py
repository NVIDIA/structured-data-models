import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import ClassShuffle, Recipe


def _labels(
    values: list[int],
    categories: tuple[str, ...] = ("a", "b", "c"),
) -> TableTensor:
    return TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            data=torch.tensor(values, dtype=torch.int32).unsqueeze(-1),
            categories=(StringTensor.from_list(categories),),
        ),
    )


def test_class_shuffle_shift_maps_labels() -> None:
    labels = _labels([0, 1, 2, -1])
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes

    output = ClassShuffle(method="shift").fit_transform(labels)

    assert output.columns[Stype.categorical] == ("target",)
    assert torch.equal(
        output.categorical.as_tensor(),
        torch.tensor([[2], [0], [1], [-1]], dtype=torch.int32),
    )
    assert output.categorical.categories[0].tolist() == ["b", "c", "a"]
    assert output.categorical.tolist() == labels.categorical.tolist()


def test_class_shuffle_random_preserves_decoded_labels() -> None:
    labels = _labels([0, 1, 2, 1])
    processor = ClassShuffle(method="random")

    transformed = processor.fit_transform(labels)

    assert transformed.categorical.tolist() == labels.categorical.tolist()


def test_class_shuffle_uses_category_count_and_preserves_missing() -> None:
    labels = _labels([0, 1, -1], categories=("a", "b", "c", "d"))

    output = ClassShuffle(method="random").fit_transform(labels)

    assert output.categorical.categories[0].numel() == 4
    assert output.categorical.as_tensor()[-1].item() == -1
    assert output.categorical.tolist() == labels.categorical.tolist()


def test_class_shuffle_learns_classes_in_recipe_target_pipeline() -> None:
    target = _labels([0, 1, 2])
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes
    recipe = Recipe(target=[ClassShuffle(method="shift")])

    transformed = recipe.target.fit_transform(target)

    assert isinstance(transformed, TableTensor)
    assert transformed.categorical.size(-1) == 1
    assert transformed.numerical.size(-1) == 0
    assert transformed.categorical.tolist() == target.categorical.tolist()
