from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest
import torch

from sdm import (
    CategoricalTensor,
    EnsembleTable,
    RelatedTables,
    Stype,
    TableTensor,
)
from sdm.processing import (
    Choice,
    Clip,
    ClipQuantiles,
    ClipSigma,
    DropConstantColumns,
    EnsembleProcessor,
    Identity,
    ImputeMean,
    PowerTransform,
    Processor,
    QuantileTransform,
    Recipe,
    ReduceEstimators,
    Sequential,
    ShuffleCategories,
    ShuffleColumns,
    Softmax,
    Standardize,
    StypeDispatch,
    TargetDecode,
    TaskDispatch,
    ToNumerical,
)


class _CenterUnlessNegative(Processor):
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        del generator
        mean = table.numerical.mean(dim=-2, keepdim=True)
        if bool((mean < 0).any()):
            raise ValueError("negative fit mean")
        self.mean = mean

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(
            numerical=table.numerical - self.mean,
        )


class _AddOneEnsemble(EnsembleProcessor):
    supported_stypes = frozenset({Stype.numerical})

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        del generator
        return table._replace_packed_representations(
            tuple(
                group.replace_blocks(numerical=group.numerical + 1)
                for group in table.iter_packed_representations()
            )
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        return table._replace_packed_representations(
            tuple(
                group.replace_blocks(numerical=group.numerical + 2)
                for group in table.iter_packed_representations()
            )
        )


class _DropsQueryRow(Processor):
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self) -> None:
        super().__init__()
        self._fit_rows = 0

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        del generator
        self._fit_rows = table.size(-2)

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.size(-2) == self._fit_rows:
            return table
        return table[..., :-1, :]


def _numerical(values: Sequence[Sequence[float]]) -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor(values, dtype=torch.float32),
        columns=tuple(f"x{i}" for i in range(len(values[0]))),
    )


def _target(*, classification: bool) -> TableTensor:
    if not classification:
        return TableTensor.from_tensor(torch.tensor([[10.0], [12.0], [14.0]]))
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0]], dtype=torch.int64),
            categories=(torch.tensor([10, 20]),),
        ),
    )


def test_ensemble_processor_scalar_fit_transform_runs_once() -> None:
    table = _numerical([[1.0], [2.0]])
    processor = _AddOneEnsemble()

    fitted = processor.fit_transform(table)
    transformed = processor.transform(table)

    torch.testing.assert_close(fitted.numerical, table.numerical + 1)
    torch.testing.assert_close(transformed.numerical, table.numerical + 2)


def test_recipe_round_robin_choice_reuses_two_variants_for_eight_members() -> (
    None
):
    features = _numerical([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    recipe = Recipe(
        features=Choice(
            Identity(),
            Clip(min_value=0.0, max_value=0.0),
            selection="round_robin",
        )
    )

    transformed, _, _ = recipe.fit_transform(
        features,
        _target(classification=False),
        num_members=8,
    )

    assert transformed.num_members == 8
    for member in range(8):
        expected = (
            features.numerical
            if member % 2 == 0
            else torch.zeros_like(features.numerical)
        )
        torch.testing.assert_close(
            transformed.representation(member).numerical,
            expected,
        )


@pytest.mark.parametrize(
    ("processor", "features", "target"),
    [
        (
            ShuffleColumns(method="shift"),
            _numerical([[1.0, 2.0], [3.0, 4.0]]),
            _target(classification=False),
        ),
        (
            ShuffleCategories(method="shift"),
            _numerical([[1.0], [2.0], [3.0]]),
            _target(classification=True),
        ),
    ],
    ids=("columns", "categories"),
)
def test_shuffle_reuses_equal_member_mappings(
    processor: Processor,
    features: TableTensor,
    target: TableTensor,
) -> None:
    recipe = (
        Recipe(features=processor)
        if isinstance(processor, ShuffleColumns)
        else Recipe(target=processor)
    )
    transformed_features, transformed_target, _ = recipe.fit_transform(
        features,
        target,
        num_members=8,
        generator=torch.Generator().manual_seed(9),
    )
    transformed = (
        transformed_features
        if isinstance(processor, ShuffleColumns)
        else transformed_target
    )

    unique = {
        (
            tuple(table.columns.items()),
            table.numerical.cpu().numpy().tobytes(),
            table.categorical.code.cpu().numpy().tobytes(),
            tuple(
                category.cpu().numpy().tobytes()
                for category in table.categorical.categories
            ),
        )
        for table in (
            transformed.representation(member) for member in range(8)
        )
    }
    assert sum(
        group.size(0) for group in transformed.iter_packed_representations()
    ) == len(unique)


def test_sampled_quantile_is_reproducible_and_member_specific() -> None:
    features = TableTensor.from_tensor(
        torch.arange(128, dtype=torch.float32).view(64, 2)
    )
    transformed = []
    for _ in range(2):
        recipe = Recipe(
            features=QuantileTransform(
                n_quantiles=4,
                subsample=10,
            )
        )
        out, _, _ = recipe.fit_transform(
            features,
            _target(classification=False),
            num_members=4,
            generator=torch.Generator().manual_seed(17),
        )
        transformed.append(out)

    for member in range(4):
        torch.testing.assert_close(
            transformed[0].representation(member).numerical,
            transformed[1].representation(member).numerical,
            rtol=0,
            atol=0,
        )
    assert any(
        not torch.equal(
            transformed[0].representation(0).numerical,
            transformed[0].representation(member).numerical,
        )
        for member in range(1, 4)
    )


def test_schema_changing_processor_splits_only_incompatible_members() -> None:
    context = _numerical(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    query = _numerical([[3.0, 1.0]])
    recipe = Recipe(
        features=Sequential(
            Choice(
                Identity(),
                Clip(min_value=0.0, max_value=0.0),
                selection="round_robin",
            ),
            DropConstantColumns(),
            Standardize(),
        )
    )

    transformed, _, _ = recipe.fit_transform(
        context,
        _target(classification=False),
        num_members=4,
    )
    query_transformed, _ = recipe.transform(query)

    assert [transformed.representation(i).size(-1) for i in range(4)] == [
        1,
        0,
        1,
        0,
    ]
    assert [
        query_transformed.representation(i).size(-1) for i in range(4)
    ] == [1, 0, 1, 0]
    assert transformed.representation(0).columns[Stype.numerical] == ("x0",)
    assert transformed.representation(1).columns[Stype.numerical] == ()
    torch.testing.assert_close(
        query_transformed.representation(0).numerical,
        torch.tensor([[2.4494898]]),
    )


def test_variable_schema_recipe_rejects_batched_logical_tables() -> None:
    recipe = Recipe(features=DropConstantColumns())

    with pytest.raises(ValueError, match=r"one logical table.*\[R, C\]"):
        recipe.fit_transform(
            TableTensor.from_tensor(torch.ones(2, 3, 1)),
            _target(classification=False),
            num_members=2,
        )


def test_nested_stype_sequential_and_choice_preserve_routes() -> None:
    table = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.categorical: ("kind",),
        },
        numerical=torch.tensor([[1.0], [2.0], [3.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0]], dtype=torch.int64),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    recipe = Recipe(
        features=StypeDispatch(
            numerical=Sequential(
                Choice(
                    Identity(),
                    Clip(min_value=0.0, max_value=0.0),
                    selection="round_robin",
                )
            ),
            categorical=Sequential(ToNumerical()),
        )
    )

    transformed, _, _ = recipe.fit_transform(
        table,
        _target(classification=True),
        num_members=2,
    )

    assert transformed.representation(0).columns[Stype.numerical] == (
        "value",
        "kind",
    )
    torch.testing.assert_close(
        transformed.representation(0).numerical,
        torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]),
    )
    torch.testing.assert_close(
        transformed.representation(1).numerical,
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    )


@pytest.mark.parametrize("classification", [False, True])
def test_nested_task_dispatch_resolves_from_target(
    classification: bool,
) -> None:
    recipe = Recipe(
        output=Sequential(
            TargetDecode(),
            ReduceEstimators(),
            StypeDispatch(
                numerical=TaskDispatch(
                    classification=Softmax(),
                    regression=Identity(),
                ),
                remainder="error",
            ),
        )
    )
    _, _, _ = recipe.fit_transform(
        _numerical([[1.0], [2.0], [3.0]]),
        _target(classification=classification),
        num_members=2,
    )
    outputs = (
        TableTensor.from_tensor(torch.tensor([[2.0, 0.0]])),
        TableTensor.from_tensor(torch.tensor([[0.0, 2.0]])),
    )
    if not classification:
        outputs = (
            TableTensor.from_tensor(torch.tensor([[10.0]])),
            TableTensor.from_tensor(torch.tensor([[14.0]])),
        )

    actual = recipe.transform_output(outputs)

    if classification:
        torch.testing.assert_close(
            actual.numerical,
            torch.tensor([[0.5, 0.5]]),
        )
    else:
        torch.testing.assert_close(actual.numerical, torch.tensor([[12.0]]))


def test_target_decode_routes_members_through_fitted_choice_groups() -> None:
    recipe = Recipe(
        target=Sequential(
            Choice(
                Identity(),
                Standardize(),
                selection="round_robin",
            ),
            Standardize(),
        ),
        output=(TargetDecode(), ReduceEstimators()),
    )
    recipe.fit_transform(
        _numerical([[1.0], [2.0], [3.0]]),
        _target(classification=False),
        num_members=8,
    )

    actual = recipe.transform_output(
        tuple(TableTensor.from_tensor(torch.tensor([[0.0]])) for _ in range(8))
    )

    torch.testing.assert_close(actual.numerical, torch.tensor([[12.0]]))


def test_related_tables_keep_table_local_fitted_state() -> None:
    features = _numerical([[0.0], [2.0], [4.0]])
    related = RelatedTables(
        tables={
            "small": _numerical([[0.0], [2.0], [4.0]]),
            "large": _numerical([[100.0], [200.0], [300.0]]),
        },
        relationships=(),
        task_links=(),
    )
    recipe = Recipe(features=Standardize())

    _, _, transformed_related = recipe.fit_transform(
        features,
        _target(classification=False),
        related,
        num_members=3,
    )

    assert transformed_related is not None
    for member in range(3):
        torch.testing.assert_close(
            transformed_related[member].tables["small"].numerical,
            torch.tensor([[-1.2247449], [0.0], [1.2247449]]),
        )
        torch.testing.assert_close(
            transformed_related[member].tables["large"].numerical,
            torch.tensor([[-1.2247449], [0.0], [1.2247449]]),
        )


def test_related_table_names_can_match_module_attributes() -> None:
    recipe = Recipe(features=Identity())
    related = RelatedTables(
        tables={"items": _numerical([[1.0], [2.0]])},
        relationships=(),
        task_links=(),
    )

    recipe.fit_transform(
        _numerical([[1.0], [2.0], [3.0]]),
        _target(classification=False),
        related,
        num_members=2,
    )
    _, transformed = recipe.transform(
        _numerical([[4.0]]),
        RelatedTables(
            tables={"items": _numerical([[3.0]])},
            relationships=(),
            task_links=(),
        ),
    )

    assert transformed is not None
    torch.testing.assert_close(
        torch.stack(
            tuple(member.tables["items"].numerical for member in transformed)
        ),
        torch.tensor([[[3.0]], [[3.0]]]),
    )


@pytest.mark.parametrize(
    "recipe",
    [
        Recipe(features=Sequential(Identity(), ReduceEstimators())),
        Recipe(target=ReduceEstimators()),
    ],
)
def test_recipe_rejects_estimator_reduction_before_output(
    recipe: Recipe,
) -> None:
    with pytest.raises(ValueError, match=r"only supported in Recipe\.output"):
        recipe.fit_transform(
            _numerical([[1.0], [2.0], [3.0]]),
            _target(classification=False),
            num_members=2,
        )


def test_target_decode_must_precede_estimator_reduction() -> None:
    recipe = Recipe(output=Sequential(ReduceEstimators(), TargetDecode()))

    with pytest.raises(ValueError, match="TargetDecode must precede"):
        recipe.fit_transform(
            _numerical([[1.0], [2.0], [3.0]]),
            _target(classification=False),
            num_members=2,
        )


@pytest.mark.parametrize(
    "output",
    [
        Sequential(TargetDecode(), TargetDecode()),
        Sequential(TargetDecode(), ReduceEstimators(), ReduceEstimators()),
    ],
)
def test_output_has_single_decode_and_reduction_boundaries(
    output: Processor,
) -> None:
    recipe = Recipe(output=output)

    with pytest.raises(ValueError, match="at most one"):
        recipe.fit_transform(
            _numerical([[1.0], [2.0], [3.0]]),
            _target(classification=False),
            num_members=2,
        )


@pytest.mark.parametrize(
    "output",
    [Standardize(), Sequential().append(Standardize())],
)
def test_output_processors_must_be_stateless(output: Processor) -> None:
    recipe = Recipe(output=output)

    with pytest.raises(
        ValueError, match="output processors must be stateless"
    ):
        recipe.fit_transform(
            _numerical([[1.0], [2.0], [3.0]]),
            _target(classification=False),
            num_members=2,
        )


@pytest.mark.parametrize(
    "processor",
    [
        Standardize(),
        ClipQuantiles(q_low=0.2, q_high=0.8),
        ClipSigma(threshold=1.5),
        PowerTransform(),
    ],
    ids=("standardize", "clip-quantiles", "clip-sigma", "power"),
)
def test_leading_variant_fit_matches_independent_scalar_fits(
    processor: Processor,
) -> None:
    fit_values = torch.tensor(
        [
            [
                [-4.0, 1.0],
                [-1.0, 2.0],
                [0.0, 4.0],
                [2.0, 8.0],
                [7.0, 16.0],
            ],
            [
                [-20.0, -3.0],
                [-10.0, -1.0],
                [0.0, 0.0],
                [10.0, 1.0],
                [20.0, 3.0],
            ],
        ]
    )
    query_values = torch.tensor(
        [
            [[-2.0, 3.0], [5.0, 12.0]],
            [[-15.0, -2.0], [15.0, 2.0]],
        ]
    )
    columns = ("left", "right")
    batched = copy.deepcopy(processor)
    batched_fit = TableTensor.from_tensor(fit_values, columns=columns)
    batched_query = TableTensor.from_tensor(query_values, columns=columns)

    actual_fit = batched.fit_transform(batched_fit)
    actual_query = batched.transform(batched_query)

    expected_fit = []
    expected_query = []
    for member in range(fit_values.size(0)):
        scalar = copy.deepcopy(processor)
        scalar_fit = TableTensor.from_tensor(
            fit_values[member],
            columns=columns,
        )
        scalar_query = TableTensor.from_tensor(
            query_values[member],
            columns=columns,
        )
        expected_fit.append(scalar.fit_transform(scalar_fit).numerical)
        expected_query.append(scalar.transform(scalar_query).numerical)

    torch.testing.assert_close(
        actual_fit.numerical,
        torch.stack(expected_fit),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        actual_query.numerical,
        torch.stack(expected_query),
        rtol=2e-5,
        atol=2e-5,
    )


def test_impute_mean_fits_each_leading_variant_independently() -> None:
    fit_values = torch.tensor(
        [
            [[1.0, float("nan")], [3.0, 4.0]],
            [[10.0, 20.0], [float("nan"), 40.0]],
        ]
    )
    query_values = torch.full((2, 1, 2), float("nan"))
    processor = ImputeMean()

    processor.fit(TableTensor.from_tensor(fit_values))
    actual = processor.transform(TableTensor.from_tensor(query_values))

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[[2.0, 4.0]], [[10.0, 30.0]]]),
    )


def test_custom_processor_contract_works_after_member_split() -> None:
    recipe = Recipe(
        features=Sequential(
            Choice(
                Identity(),
                Clip(min_value=0.0, max_value=0.0),
                selection="round_robin",
            ),
            _CenterUnlessNegative(),
        )
    )

    features = _numerical([[1.0], [2.0], [3.0]])
    transformed, _, _ = recipe.fit_transform(
        features,
        _target(classification=False),
        num_members=2,
    )

    torch.testing.assert_close(
        transformed.representation(0).numerical,
        torch.tensor([[-1.0], [0.0], [1.0]]),
    )
    torch.testing.assert_close(
        transformed.representation(1).numerical,
        torch.zeros_like(features.numerical),
    )


def test_recipe_rejects_mixed_execution_devices() -> None:
    target = TableTensor.from_tensor(torch.ones(3, 1, device="meta"))

    with pytest.raises(ValueError, match="same device"):
        Recipe().fit_transform(
            _numerical([[1.0], [2.0], [3.0]]),
            target,
            num_members=2,
        )


def test_recipe_rejects_row_changes_during_query_transform() -> None:
    recipe = Recipe(features=_DropsQueryRow())
    recipe.fit_transform(
        _numerical([[1.0], [2.0], [3.0]]),
        _target(classification=False),
        num_members=2,
    )

    with pytest.raises(ValueError, match="row dimension"):
        recipe.transform(_numerical([[4.0], [5.0]]))


def test_failed_refit_keeps_the_previous_complete_recipe_plan() -> None:
    recipe = Recipe(features=_CenterUnlessNegative())
    first_context = _numerical([[1.0], [3.0]])
    query = _numerical([[5.0]])

    recipe.fit_transform(
        first_context,
        _target(classification=False),
        num_members=2,
    )
    before, _ = recipe.transform(query)

    with pytest.raises(ValueError, match="negative fit mean"):
        recipe.fit_transform(
            _numerical([[-3.0], [-1.0]]),
            _target(classification=False),
            num_members=2,
        )

    after, _ = recipe.transform(query)
    for member in range(2):
        torch.testing.assert_close(
            before.representation(member).numerical,
            torch.tensor([[3.0]]),
        )
        torch.testing.assert_close(
            after.representation(member).numerical,
            before.representation(member).numerical,
            rtol=0,
            atol=0,
        )
