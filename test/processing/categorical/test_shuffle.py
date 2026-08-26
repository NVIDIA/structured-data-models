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
    if numerical is not None:
        numerical = numerical.to(device)
    return TableTensor(
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

    assert output.columns[Stype.categorical] == ("cat_0",)
    assert torch.equal(
        output.categorical.code[:3, 0].sort().values,
        torch.arange(3, dtype=torch.int32),
    )
    assert output.categorical.code[-1].item() == -1
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

    valid_mask = features.categorical.isfinite()
    assert torch.equal(
        transformed.categorical.code[~valid_mask],
        features.categorical.code[~valid_mask],
    )
    assert transformed.categorical.code.device == device
    for transformed_category, category in zip(
        transformed.categorical.categories,
        features.categorical.categories,
        strict=True,
    ):
        assert sorted(transformed_category.tolist()) == sorted(
            category.tolist()
        )
    assert transformed.categorical.tolist() == features.categorical.tolist()


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_categories_is_reproducible_with_generator(
    method: Literal["shift", "random"],
) -> None:
    features = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
    )

    first = ShuffleCategories(method=method).fit_transform(
        features,
        generator=torch.Generator().manual_seed(0),
    )
    second = ShuffleCategories(method=method).fit_transform(
        features,
        generator=torch.Generator().manual_seed(0),
    )

    assert first.equal(second)


def test_shuffle_categories_preserves_missing() -> None:
    target = _table(
        [[0], [1], [-1]],
        (("a", "b", "c", "d"),),
    )

    processor = ShuffleCategories(method="random")
    output = processor.fit_transform(target)

    assert output.categorical.categories[0].numel() == 4
    assert output.categorical.code[-1].item() == -1
    assert output.categorical.tolist() == target.categorical.tolist()


def test_shuffle_categories_preserves_empty_vocabulary() -> None:
    target = _table(
        [[-1], [-1]],
        ((),),
    )

    output = ShuffleCategories(method="random").fit_transform(target)

    assert output.equal(target)


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


@pytest.mark.parametrize("method", ["shift", "random"])
@withCUDA
def test_shuffle_categories_shares_permutations_by_table_repetition(
    method: Literal["shift", "random"],
    device: torch.device,
) -> None:
    first = _table(
        [[0], [1], [2]],
        (("a", "b", "c"),),
        device=device,
    )
    second = _table(
        [[0], [1], [2]],
        (("x", "y", "z"),),
        device=device,
    )
    ensemble = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )
    generator = torch.Generator(device=device).manual_seed(7)

    processor = ShuffleCategories(method=method).fit_ensemble(
        ensemble,
        generator=generator,
    )
    output = processor.transform_ensemble(ensemble)

    assert torch.equal(
        output.table(0).categorical.code,
        output.table(1).categorical.code,
    )
    assert torch.equal(
        output.table(2).categorical.code,
        output.table(3).categorical.code,
    )
    for member_id in range(ensemble.num_members):
        assert output.table(member_id).categorical.tolist() == (
            ensemble.table(member_id).categorical.tolist()
        )


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_categories_uses_schema_compatible_shared_states(
    method: Literal["shift", "random"],
) -> None:
    first = _table(
        [[0], [1], [2]],
        (("a", "b", "c"),),
    )
    second = _table(
        [[0, 0], [1, 1], [2, 0]],
        (("x", "y", "z"), ("off", "on")),
    )
    ensemble = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )

    processor = ShuffleCategories(method=method).fit_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(7),
    )
    output = processor.transform_ensemble(ensemble)

    assert output.table(0).categorical.size(-1) == 1
    assert output.table(1).categorical.size(-1) == 2
    for member_id in range(ensemble.num_members):
        assert output.table(member_id).categorical.tolist() == (
            ensemble.table(member_id).categorical.tolist()
        )


@pytest.mark.parametrize("method", ["shift", "random"])
@withCUDA
def test_shuffle_categories_restores_ensemble_state(
    method: Literal["shift", "random"],
    device: torch.device,
) -> None:
    first = _table(
        [[0], [1], [2]],
        (("a", "b", "c"),),
        device=device,
    )
    second = _table(
        [[2], [1], [0]],
        (("x", "y", "z"),),
        device=device,
    )
    ensemble = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )
    processor = ShuffleCategories(method=method).fit_ensemble(
        ensemble,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    expected = processor.transform_ensemble(ensemble)

    restored = ShuffleCategories(method=method)
    restored.load_state_dict(processor.state_dict())
    output = restored.transform_ensemble(ensemble)

    assert restored.is_fitted
    for member_id in range(ensemble.num_members):
        assert output.table(member_id).equal(expected.table(member_id))


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
