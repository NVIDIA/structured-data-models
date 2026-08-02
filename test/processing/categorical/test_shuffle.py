from typing import Literal

import pytest
import torch

from sdm import (
    CategoricalTensor,
    EnsembleTable,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import EnsembleProcessor, ShuffleCategories
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
            code=torch.tensor(values, dtype=torch.int32, device=device),
            categories=tuple(
                StringTensor.from_list(category, device=device)
                for category in categories
            ),
        ),
    )


def test_shuffle_categories_shift_maps_single_target() -> None:
    target = _table(
        [[0], [1], [2], [-1]],
        (("a", "b", "c"),),
    )
    processor = ShuffleCategories(method="shift")

    output = processor.fit_transform(target)

    assert output.columns[Stype.categorical] == ("cat0",)
    permutation = processor.permutations
    codes = target.categorical.code
    valid = target.categorical.isfinite()
    assert torch.equal(
        output.categorical.code[valid],
        permutation[codes[valid].to(torch.long)].to(codes.dtype),
    )
    assert output.categorical.code[-1].item() == -1
    assert (
        output.categorical.categories[0].tolist()
        == target.categorical.categories[0][permutation.argsort()].tolist()
    )
    assert output.categorical.tolist() == target.categorical.tolist()


@withCUDA
def test_shuffle_categories_random_permutes_each_categorical_column(
    device: torch.device,
) -> None:
    features = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
        device=device,
    )
    processor = ShuffleCategories(method="random")

    transformed = processor.fit_transform(features)

    offsets = processor.offsets.tolist()
    input_codes = features.categorical.code
    output_codes = transformed.categorical.code
    valid_mask = features.categorical.isfinite()
    for index, category in enumerate(features.categorical.categories):
        permutation = processor.permutations[
            offsets[index] : offsets[index + 1]
        ]
        assert torch.equal(
            permutation.sort().values,
            torch.arange(category.numel(), device=device),
        )
        valid = valid_mask[..., index]
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


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_categories_is_reproducible_with_generator(
    method: Literal["shift", "random"],
) -> None:
    features = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
    )

    first = ShuffleCategories(method=method).fit(
        features,
        generator=torch.Generator().manual_seed(0),
    )
    second = ShuffleCategories(method=method).fit(
        features,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(first.permutations, second.permutations)


def test_shuffle_categories_preserves_missing() -> None:
    target = _table(
        [[0], [1], [-1]],
        (("a", "b", "c", "d"),),
    )

    processor = ShuffleCategories(method="random")
    output = processor.fit_transform(target)

    assert processor.offsets.tolist() == [0, 4]
    assert processor.permutations.numel() == 4
    assert output.categorical.categories[0].numel() == 4
    assert output.categorical.code[-1].item() == -1
    assert output.categorical.tolist() == target.categorical.tolist()


def test_shuffle_categories_is_an_ensemble_processor() -> None:
    assert issubclass(ShuffleCategories, EnsembleProcessor)


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_categories_ensemble_matches_independent_processors(
    method: Literal["shift", "random"],
) -> None:
    context = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
    )
    query = _table(
        [[2, 1], [0, -1]],
        (("a", "b", "c"), ("x", "y")),
    )
    processor = ShuffleCategories(method=method)

    context_output = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=8),
        generator=torch.Generator().manual_seed(7),
    )
    query_output = processor.transform_ensemble(
        EnsembleTable(query, num_members=8)
    )

    generator = torch.Generator().manual_seed(7)
    references = [ShuffleCategories(method=method) for _ in range(8)]
    for member_id, reference in enumerate(references):
        expected_context = reference.fit_transform(
            context,
            generator=generator,
        )
        expected_query = reference.transform(query)
        assert context_output.representation(member_id).equal(expected_context)
        assert query_output.representation(member_id).equal(expected_query)


def test_shuffle_categories_reuses_equal_member_permutations() -> None:
    table = _table([[0], [1]], (("a", "b"),))
    output = ShuffleCategories(method="shift").fit_transform_ensemble(
        EnsembleTable(table, num_members=8),
        generator=torch.Generator().manual_seed(9),
    )

    unique_codes = {
        tuple(
            output.representation(member_id)
            .categorical.code.flatten()
            .tolist()
        )
        for member_id in range(output.num_members)
    }
    assert sum(
        packed.size(0) for packed in output.iter_packed_representations()
    ) == len(unique_codes)
