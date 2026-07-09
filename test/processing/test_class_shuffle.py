import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import ClassShuffle, Recipe, StypeDispatch
from sdm.testing import withCUDA


def _table(
    values: list[list[int]],
    categories: tuple[tuple[str, ...], ...],
    numerical: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> TableTensor:
    columns: dict[str, tuple[str, ...]] = {
        "categorical": tuple(f"cat{i}" for i in range(len(categories))),
    }
    if numerical is not None:
        columns["numerical"] = tuple(
            f"num{i}" for i in range(numerical.size(-1))
        )
        numerical = numerical.to(device)
    return TableTensor(
        columns=columns,
        numerical=numerical,
        categorical=CategoricalTensor(
            data=torch.tensor(values, dtype=torch.int32, device=device),
            categories=tuple(
                StringTensor.from_list(category, device=device)
                for category in categories
            ),
        ),
    )


def test_class_shuffle_shift_maps_single_target() -> None:
    target = _table(
        [[0], [1], [2], [-1]],
        (("a", "b", "c"),),
    )
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes
    processor = ClassShuffle(method="shift")

    output = processor.fit_transform(target)

    assert output.columns[Stype.categorical] == ("cat0",)
    permutation = processor.permutations
    codes = target.categorical.as_tensor()
    valid = codes >= 0
    assert torch.equal(
        output.categorical.as_tensor()[valid],
        permutation[codes[valid].to(torch.long)].to(codes.dtype),
    )
    assert output.categorical.as_tensor()[-1].item() == -1
    assert (
        output.categorical.categories[0].tolist()
        == target.categorical.categories[0][permutation.argsort()].tolist()
    )
    assert output.categorical.tolist() == target.categorical.tolist()


@withCUDA
def test_class_shuffle_random_permutes_each_categorical_column(
    device: torch.device,
) -> None:
    features = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
        device=device,
    )
    torch.manual_seed(0)
    processor = ClassShuffle(method="random")

    transformed = processor.fit_transform(features)

    offsets = processor.offsets.tolist()
    input_codes = features.categorical.as_tensor()
    output_codes = transformed.categorical.as_tensor()
    for index, category in enumerate(features.categorical.categories):
        permutation = processor.permutations[
            offsets[index] : offsets[index + 1]
        ]
        assert torch.equal(
            permutation.sort().values,
            torch.arange(category.numel(), device=device),
        )
        valid = input_codes[..., index] >= 0
        assert torch.equal(
            output_codes[..., index][valid],
            permutation[input_codes[..., index][valid].to(torch.long)].to(
                input_codes.dtype
            ),
        )
        assert torch.equal(
            output_codes[..., index][~valid],
            input_codes[..., index][~valid],
        )
        assert (
            transformed.categorical.categories[index].tolist()
            == category[permutation.argsort()].tolist()
        )

    assert processor.permutations.device == device
    assert processor.offsets.device == device
    assert transformed.categorical.tolist() == features.categorical.tolist()


def test_class_shuffle_uses_category_count_and_preserves_missing() -> None:
    target = _table(
        [[0], [1], [-1]],
        (("a", "b", "c", "d"),),
    )

    processor = ClassShuffle(method="random")
    output = processor.fit_transform(target)

    assert processor.offsets.tolist() == [0, 4]
    assert processor.permutations.numel() == 4
    assert output.categorical.categories[0].numel() == 4
    assert output.categorical.as_tensor()[-1].item() == -1
    assert output.categorical.tolist() == target.categorical.tolist()


def test_class_shuffle_runs_on_mixed_feature_blocks() -> None:
    numerical = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    features = _table(
        [[0, 1], [1, 0], [2, -1]],
        (("a", "b", "c"), ("x", "y")),
        numerical=numerical,
    )
    torch.manual_seed(0)
    recipe = Recipe(
        features=[
            StypeDispatch(
                categorical=ClassShuffle(method="random"),
            )
        ]
    )

    transformed = recipe.features.fit_transform(features)

    assert transformed.columns == features.columns
    assert torch.equal(transformed.numerical, numerical)
    assert transformed.categorical.tolist() == features.categorical.tolist()
    assert not torch.equal(
        transformed.categorical.as_tensor(),
        features.categorical.as_tensor(),
    )


def test_class_shuffle_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method"):
        ClassShuffle(method="invalid")  # type: ignore


def test_class_shuffle_rejects_transform_schema_mismatch() -> None:
    fitted = _table(
        [[0, 0], [1, 1]],
        (("a", "b"), ("x", "y", "z")),
    )
    processor = ClassShuffle().fit(fitted)

    fewer_columns = _table(
        [[0], [1]],
        (("a", "b"),),
    )
    with pytest.raises(ValueError, match="fitted column count"):
        processor.transform(fewer_columns)

    changed_cardinalities = _table(
        [[0, 0], [1, 1]],
        (("a", "b", "c"), ("x", "y")),
    )
    with pytest.raises(ValueError, match=r"'cat0'.*category count"):
        processor.transform(changed_cardinalities)


def test_class_shuffle_state_dict_round_trip() -> None:
    table = _table(
        [[0, 0], [1, 1], [2, -1]],
        (("a", "b", "c"), ("x", "y")),
    )
    torch.manual_seed(0)
    fitted = ClassShuffle(method="random").fit(table)
    restored = ClassShuffle(method="random")

    restored.load_state_dict(fitted.state_dict())

    assert torch.equal(restored.permutations, fitted.permutations)
    assert torch.equal(restored.offsets, fitted.offsets)
    assert torch.equal(
        restored.transform(table).categorical.as_tensor(),
        fitted.transform(table).categorical.as_tensor(),
    )
