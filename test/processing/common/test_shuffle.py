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


def test_shuffle_columns_defaults_to_random() -> None:
    assert ShuffleColumns().method == "random"


def test_shuffle_columns_shift_rotates_numerical_block() -> None:
    table = _table()

    output = ShuffleColumns(method="shift").fit_transform(
        table,
        generator=torch.Generator().manual_seed(3),
    )

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("1", "2", "0")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([1, 2, 0])),
    )


@pytest.mark.parametrize("method", ["shift", "random", "latin"])
def test_shuffle_columns_scalar_fit_transform_and_inverse(
    method: Literal["shift", "random", "latin"],
) -> None:
    table = _table()
    processor = ShuffleColumns(method=method)

    processor.fit(table)
    transformed = processor.transform(table)
    restored = processor.inverse_transform(transformed)

    assert restored.equal(table)


@pytest.mark.parametrize("method", ["shift", "random", "latin"])
def test_shuffle_columns_is_reproducible_with_generator(
    method: Literal["shift", "random", "latin"],
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


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_columns_ensemble_matches_independent_processors(
    method: Literal["shift", "random"],
) -> None:
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


def test_shuffle_columns_checks_num_members() -> None:
    processor = ShuffleColumns(method="shift")
    processor.fit_ensemble(
        EnsembleTable(_table(), num_members=8),
        generator=torch.Generator().manual_seed(9),
    )

    with pytest.raises(
        RuntimeError,
        match="was fitted with 8 ensemble members, but got 7",
    ):
        processor.transform_ensemble(EnsembleTable(_table(), num_members=7))


def _wide_table(
    n_features: int,
    *,
    device: torch.device,
    prefix: str = "x",
) -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(n_features, dtype=torch.float32, device=device)[None],
        columns=tuple(f"{prefix}{index}" for index in range(n_features)),
    )


@withCUDA
@pytest.mark.parametrize(
    ("n_features", "num_members"),
    [(2, 5), (7, 14), (17, 8), (4001, 128)],
)
def test_shuffle_columns_latin_invariants(
    n_features: int,
    num_members: int,
    device: torch.device,
) -> None:
    table = _wide_table(n_features, device=device)
    output = ShuffleColumns(method="latin").fit_transform_ensemble(
        EnsembleTable(table, num_members=num_members),
        generator=torch.Generator(device=device).manual_seed(7),
    )
    columns = set(table.columns[Stype.numerical])
    orders = [
        output.table(member_id).columns[Stype.numerical]
        for member_id in range(num_members)
    ]

    assert all(set(order) == columns for order in orders)
    for start in range(0, num_members, n_features):
        cycle = orders[start : start + n_features]
        for position in range(n_features):
            assert len({order[position] for order in cycle}) == len(cycle)


@withCUDA
def test_shuffle_columns_latin_schemas_and_state_dict(
    device: torch.device,
) -> None:
    first = _wide_table(4, device=device, prefix="a")
    second = TableTensor.from_tensor(
        torch.arange(3, dtype=torch.float32, device=device)[None],
        columns=("a1", "a2", "b0"),
    )
    member_table_ids = (0, 1, 0, 1, 1, 0)
    ensemble = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=member_table_ids,
    )
    processor = ShuffleColumns(method="latin")
    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator(device=device).manual_seed(11),
    )

    restored = ShuffleColumns(method="latin")
    restored.load_state_dict(processor.state_dict())
    again = restored.transform_ensemble(ensemble)
    inverse = restored.inverse_transform_ensemble(again)

    assert restored.is_fitted
    assert all(
        output.table(member_id).equal(again.table(member_id))
        for member_id in range(ensemble.num_members)
    )
    assert all(
        inverse.table(member_id).equal(ensemble.table(member_id))
        for member_id in range(ensemble.num_members)
    )
    assert all(group.device == device for group in again)


def _reference_latin(
    n_features: int,
    num_classes: int,
    num_members: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)

    def square(symbols: list[int]) -> list[list[int]]:
        if len(symbols) == 1:
            return [symbols]
        symbol = rng.choice(symbols)
        symbols.remove(symbol)
        rows = square(symbols)
        rows.append(rows[0].copy())
        for index, row in enumerate(rows):
            row.insert(index, symbol)
        return rows

    rows = square(list(range(n_features)))
    rng.shuffle(rows)
    features = list(map(list, zip(*rows, strict=True)))
    rng.shuffle(features)
    pairs = [
        (feature, class_id)
        for feature in features
        for class_id in range(num_classes)
    ]
    random.Random(seed).shuffle(pairs)
    return [feature for pair in pairs for feature in (pair[0], pair[0])][
        :num_members
    ]


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n_features", [2, 7, 100])
@pytest.mark.parametrize("num_classes", [1, 3])
def test_tabiclv2_latin_plan_matches_reference(
    seed: int,
    n_features: int,
    num_classes: int,
) -> None:
    num_members = min(8, 2 * n_features * num_classes)
    plan = _TabICLv2EstimatorPlan()
    plan._num_classes = num_classes
    plan._num_members = num_members
    plan._seed = seed

    processor = _TabICLv2ShuffleColumns(plan)
    processor.fit_ensemble(
        EnsembleTable(
            _wide_table(n_features, device=torch.device("cpu")),
            num_members=num_members,
        )
    )
    actual = processor._permutations[0].tolist()

    assert actual == _reference_latin(
        n_features,
        num_classes,
        num_members,
        seed,
    )


def test_tabiclv2_latin_plan_single_member_is_reference_identity() -> None:
    plan = _TabICLv2EstimatorPlan()
    processor = _TabICLv2ShuffleColumns(plan)

    processor.fit_ensemble(
        EnsembleTable(
            _wide_table(7, device=torch.device("cpu")),
            num_members=1,
        )
    )

    assert processor._permutations[0].tolist() == [list(range(7))]


def test_tabiclv2_latin_plan_scales_past_reference_limit() -> None:
    plan = _TabICLv2EstimatorPlan()
    plan._num_members = 8
    plan._seed = 42
    processor = _TabICLv2ShuffleColumns(plan)
    processor.fit_ensemble(
        EnsembleTable(
            _wide_table(4001, device=torch.device("cpu")),
            num_members=8,
        )
    )
    permutations = processor._permutations[0]

    assert permutations.size() == (8, 4001)
    assert torch.equal(permutations[::2], permutations[1::2])
    assert all(
        torch.unique(permutations[::2, position]).numel() == 4
        for position in (0, 2000, 4000)
    )


def test_tabiclv2_latin_plan_checks_estimator_capacity() -> None:
    plan = _TabICLv2EstimatorPlan()
    plan._num_members = 3

    with pytest.raises(ValueError, match="supports at most 2 members"):
        plan.latin_state(1, tuple(range(3)), torch.device("cpu"))


@pytest.mark.parametrize("seed", [0, 1, 42, 1729])
def test_tabiclv2_latin_plan_matches_rng_at_4000(seed: int) -> None:
    rng = random.Random(seed)
    remaining = list(range(4000))
    base = []
    while len(remaining) > 1:
        base.append(remaining.pop(rng.randrange(len(remaining))))
    base.extend(remaining)
    rows = list(range(4000))
    rng.shuffle(rows)
    patterns = list(range(4000))
    rng.shuffle(patterns)

    assert _TabICLv2EstimatorPlan._draw_latin(4000, seed) == (
        base,
        rows,
        patterns,
    )


@pytest.mark.parametrize("classification", [False, True])
def test_tabiclv2_recipe_uses_reference_estimator_mappings(
    classification: bool,
) -> None:
    n_features = 7
    features = TableTensor.from_tensor(
        torch.arange(56, dtype=torch.float32).reshape(8, n_features),
        columns=tuple(f"x{index}" for index in range(n_features)),
    )
    if classification:
        target = TableTensor(
            columns={Stype.categorical: ("target",)},
            categorical=CategoricalTensor(
                code=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])[:, None],
                categories=(torch.tensor([0, 1, 2, 99]),),
            ),
        )
        num_classes = 3
    else:
        target = TableTensor.from_tensor(
            torch.arange(8, dtype=torch.float32)[:, None]
        )
        num_classes = 1

    contexts = RecipeExecution(default_recipe()).fit_transform(
        features,
        target,
        None,
        num_members=8,
        generator=torch.Generator().manual_seed(42),
    )
    actual = [
        [
            int(column.removeprefix("x"))
            for column in context.x.columns[Stype.numerical]
        ]
        for context in contexts
    ]

    assert actual == _reference_latin(
        n_features,
        num_classes,
        8,
        42,
    )
