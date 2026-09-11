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
from sdm.processing import ShuffleCategories
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
    processor.fit_ensemble(EnsembleTable.from_table(table, num_members=2))

    with pytest.raises(RuntimeError, match="same number"):
        processor.transform_ensemble(
            EnsembleTable.from_table(table, num_members=1)
        )


def test_shuffle_categories_refit_replaces_ensemble_state() -> None:
    table = _table(
        [[0, 0], [1, 1], [2, -1], [1, 0]],
        (("a", "b", "c"), ("x", "y")),
    )
    processor = ShuffleCategories(method="random").fit_ensemble(
        EnsembleTable.from_table(table, num_members=2),
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


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_categories_preserves_strided_input_and_missing_codes(
    device: torch.device,
    dtype: torch.dtype,
    method: Literal["shift", "random"],
) -> None:
    values = torch.tensor(
        [[0, -7, 1], [2, -1, -3], [-2, -9, 0]],
        dtype=dtype,
        device=device,
    ).t()
    values = values.contiguous().t()
    categories = (
        torch.arange(3, device=device),
        torch.empty(0, device=device),
        torch.arange(2, device=device),
    )
    table = TableTensor(
        categorical=CategoricalTensor(code=values, categories=categories)
    )
    original = values.clone()
    processor = ShuffleCategories(method=method).fit(table)

    first = processor.transform(table)
    second = processor.transform(table)

    assert torch.equal(values, original)
    assert first.equal(second)
    assert first.categorical.code.dtype == dtype
    missing = original < 0
    assert torch.equal(first.categorical.code[missing], original[missing])
    assert (
        first.categorical[..., [0, 2]].tolist()
        == table.categorical[..., [0, 2]].tolist()
    )
    first.categorical.code.fill_(-1)
    assert torch.equal(values, original)
    assert processor.transform(table).equal(second)
