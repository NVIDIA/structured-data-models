import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import CategoryShuffle
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


def test_category_shuffle_shift_maps_single_target() -> None:
    target = _table(
        [[0], [1], [2], [-1]],
        (("a", "b", "c"),),
    )
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three classes
    processor = CategoryShuffle(method="shift")

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
def test_category_shuffle_random_permutes_each_categorical_column(
    device: torch.device,
) -> None:
    features = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
        device=device,
    )
    torch.manual_seed(0)
    processor = CategoryShuffle(method="random")

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


def test_category_shuffle_uses_category_count_and_preserves_missing() -> None:
    target = _table(
        [[0], [1], [-1]],
        (("a", "b", "c", "d"),),
    )

    processor = CategoryShuffle(method="random")
    output = processor.fit_transform(target)

    assert processor.offsets.tolist() == [0, 4]
    assert processor.permutations.numel() == 4
    assert output.categorical.categories[0].numel() == 4
    assert output.categorical.as_tensor()[-1].item() == -1
    assert output.categorical.tolist() == target.categorical.tolist()


@withCUDA
def test_category_shuffle_inverse_restores_class_scores(
    device: torch.device,
) -> None:
    target = _table(
        [[0], [1], [2], [3]],
        (("a", "b", "c", "d"),),
        device=device,
    )
    torch.manual_seed(1)
    processor = CategoryShuffle(method="random").fit(target)
    permutation = processor.permutations
    assert not torch.equal(permutation, permutation.argsort())

    original = torch.arange(
        2 * 3 * 4,
        dtype=torch.float64,
        device=device,
    ).reshape(2, 3, 4)
    model_output = torch.full(
        (2, 3, 10),
        -100.0,
        dtype=original.dtype,
        device=device,
    )
    model_output[..., permutation] = original

    restored = processor.inverse_transform(
        TableTensor.from_tensor(model_output)
    )

    assert restored.size() == original.size()
    assert restored.numerical.dtype == original.dtype
    assert restored.device == device
    assert restored.categorical.size(-1) == 0
    torch.testing.assert_close(
        restored.numerical,
        original,
        rtol=0,
        atol=0,
    )


def test_category_shuffle_inverse_requires_single_categorical_target() -> None:
    features = _table(
        [[0, 0], [1, 1]],
        (("a", "b"), ("x", "y")),
    )
    processor = CategoryShuffle(method="random").fit(features)
    model_output = TableTensor.from_tensor(torch.randn(2, 10))

    with pytest.raises(ValueError, match="exactly one categorical column"):
        processor.inverse_transform(model_output)
