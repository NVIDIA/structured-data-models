import random
from typing import Literal

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models.tabiclv2.recipe import (
    _TabICLv2EstimatorPlan,
    _TabICLv2ShuffleColumns,
    default_recipe,
)
from sdm.processing import ShuffleColumns
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )


@pytest.mark.parametrize("method", ["random", "latin"])
def test_shuffle_columns_scalar_fit_transform_and_inverse(
    method: Literal["random", "latin"],
) -> None:
    table = _table()
    processor = ShuffleColumns(method=method)

    processor.fit(table)
    transformed = processor.transform(table)
    restored = processor.inverse_transform(transformed)

    assert restored.equal(table)


@pytest.mark.parametrize("method", ["random", "latin"])
def test_shuffle_columns_is_reproducible_with_generator(
    method: Literal["random", "latin"],
) -> None:
    table = _table()

    first = ShuffleColumns(method=method)
    first_output = first.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = ShuffleColumns(method=method)
    second_output = second.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert first_output.equal(second_output)

@pytest.mark.parametrize("method", ["random", "latin"])
def test_random_ensemble_matches_independent_shuffles(method: Literal["random", "latin"]) -> None:
    context = _table()
    query = context.replace_blocks(numerical=context.numerical + 10)
    ensemble = ShuffleColumns(method=method)
    ensemble_generator = torch.Generator().manual_seed(7)

    context_output = ensemble.fit_transform_ensemble(
        EnsembleTable(context, num_members=8),
        generator=ensemble_generator,
    )
    query_output = ensemble.transform_ensemble(
        EnsembleTable(query, num_members=8)
    )
    restored = ensemble.inverse_transform_ensemble(context_output)

    reference_generator = torch.Generator().manual_seed(7)
    references = [ShuffleColumns(method=method) for _ in range(8)]
    for member_id, processor in enumerate(references):
        expected_context = processor.fit_transform(
            context,
            generator=reference_generator,
        )
        expected_query = processor.transform(query)
        assert context_output.table(member_id).equal(expected_context)
        assert query_output.table(member_id).equal(expected_query)
        assert restored.table(member_id).equal(context)


@pytest.mark.parametrize(
    "method_name",
    ["transform_ensemble", "inverse_transform_ensemble"],
)
def test_shuffle_columns_checks_num_members(method_name: str) -> None:
    processor = ShuffleColumns(method="latin")
    processor.fit_ensemble(
        EnsembleTable(_table(), num_members=8),
        generator=torch.Generator().manual_seed(9),
    )

    with pytest.raises(
        RuntimeError,
        match="was fitted with 8 ensemble members, but got 7",
    ):
        getattr(processor, method_name)(EnsembleTable(_table(), num_members=7))
