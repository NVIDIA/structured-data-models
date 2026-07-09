import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import ClassShuffle, Recipe


def _table(
    values: list[list[int]],
    categories: tuple[tuple[str, ...], ...],
    numerical: torch.Tensor | None = None,
) -> TableTensor:
    columns: dict[str, tuple[str, ...]] = {
        "categorical": tuple(f"cat{i}" for i in range(len(categories))),
    }
    if numerical is not None:
        columns["numerical"] = tuple(
            f"num{i}" for i in range(numerical.size(-1))
        )
    return TableTensor(
        columns=columns,
        numerical=numerical,
        categorical=CategoricalTensor(
            data=torch.tensor(values, dtype=torch.int32),
            categories=tuple(
                StringTensor.from_list(category) for category in categories
            ),
        ),
    )


def test_class_shuffle_shift_maps_single_target() -> None:
    target = _table(
        [[0], [1], [2], [-1]],
        (("a", "b", "c"),),
    )
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes

    output = ClassShuffle(method="shift").fit_transform(target)

    assert output.columns[Stype.categorical] == ("cat0",)
    assert torch.equal(
        output.categorical.as_tensor(),
        torch.tensor([[2], [0], [1], [-1]], dtype=torch.int32),
    )
    assert output.categorical.categories[0].tolist() == ["b", "c", "a"]
    assert output.categorical.tolist() == target.categorical.tolist()


def test_class_shuffle_random_permutes_each_categorical_column() -> None:
    features = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
    )
    processor = ClassShuffle(method="random")

    transformed = processor.fit_transform(features)

    assert processor.offsets.tolist() == [0, 3, 5]
    assert processor.permutations.numel() == 5
    assert transformed.categorical.tolist() == features.categorical.tolist()
    assert transformed.categorical.as_tensor()[-2, 1].item() == -1


def test_class_shuffle_uses_category_count_and_preserves_missing() -> None:
    target = _table(
        [[0], [1], [-1]],
        (("a", "b", "c", "d"),),
    )

    output = ClassShuffle(method="random").fit_transform(target)

    assert output.categorical.categories[0].numel() == 4
    assert output.categorical.as_tensor()[-1].item() == -1
    assert output.categorical.tolist() == target.categorical.tolist()


def test_class_shuffle_runs_in_recipe_target_pipeline() -> None:
    target = _table(
        [[0], [1], [2]],
        (("a", "b", "c"),),
    )
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes
    recipe = Recipe(target=[ClassShuffle(method="shift")])

    transformed = recipe.target.fit_transform(target)

    assert isinstance(transformed, TableTensor)
    assert transformed.categorical.size(-1) == 1
    assert transformed.numerical.size(-1) == 0
    assert transformed.categorical.tolist() == target.categorical.tolist()


def test_class_shuffle_runs_on_mixed_feature_blocks() -> None:
    numerical = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    features = _table(
        [[0, 1], [1, 0], [2, -1]],
        (("a", "b", "c"), ("x", "y")),
        numerical=numerical,
    )
    torch.manual_seed(3)
    recipe = Recipe(features=[ClassShuffle(method="random")])

    transformed = recipe.features.fit_transform(features)

    assert transformed.columns == features.columns
    assert torch.equal(transformed.numerical, numerical)
    assert transformed.categorical.tolist() == features.categorical.tolist()
    assert not torch.equal(
        transformed.categorical.as_tensor(),
        features.categorical.as_tensor(),
    )
