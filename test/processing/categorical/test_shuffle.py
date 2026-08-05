from typing import Literal

import pytest
import torch

from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import ShuffleCategories
from sdm.tensor import EnsembleTable
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


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_categories_ensemble_matches_independent_processors(
    method: Literal["shift", "random"],
) -> None:
    first_context = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
    )
    second_context = _table(
        [[0], [1], [3], [2]],
        (("w", "x", "y", "z"),),
    )
    first_query = _table(
        [[2, 1], [0, -1]],
        (("a", "b", "c"), ("x", "y")),
    )
    second_query = _table(
        [[3], [0]],
        (("w", "x", "y", "z"),),
    )
    table_ids = (1, 0, 1, 0, 0, 1, 0, 1)
    context = EnsembleTable.from_tables(
        tables=(first_context, second_context),
        member_table_ids=table_ids,
    )
    query = EnsembleTable.from_tables(
        tables=(first_query, second_query),
        member_table_ids=table_ids,
    )
    processor = ShuffleCategories(method=method)

    context_output = processor.fit_transform_ensemble(
        context,
        generator=torch.Generator().manual_seed(7),
    )
    query_output = processor.transform_ensemble(query)

    generator = torch.Generator().manual_seed(7)
    context_tables = (first_context, second_context)
    query_tables = (first_query, second_query)
    for member_id, table_id in enumerate(table_ids):
        reference = ShuffleCategories(method=method)
        expected_context = reference.fit_transform(
            context_tables[table_id],
            generator=generator,
        )
        expected_query = reference.transform(query_tables[table_id])
        assert context_output.table(member_id).equal(expected_context)
        assert query_output.table(member_id).equal(expected_query)


def test_shuffle_categories_requires_fitted_member_count() -> None:
    table = _table([[0], [1]], (("a", "b"),))
    processor = ShuffleCategories(method="shift")
    processor.fit_ensemble(EnsembleTable(table, num_members=2))

    with pytest.raises(RuntimeError, match="same number"):
        processor.transform_ensemble(EnsembleTable(table, num_members=1))


def test_shuffle_categories_refit_replaces_ensemble_state() -> None:
    table = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
    )
    processor = ShuffleCategories(method="random").fit_ensemble(
        EnsembleTable(table, num_members=2),
        generator=torch.Generator().manual_seed(0),
    )
    output = processor.fit_transform(
        table,
        generator=torch.Generator().manual_seed(1),
    )
    reference = ShuffleCategories(method="random")
    expected = reference.fit_transform(
        table,
        generator=torch.Generator().manual_seed(1),
    )

    assert output.equal(expected)
    assert torch.equal(processor.permutations, reference.permutations)
    assert torch.equal(processor.offsets, reference.offsets)
