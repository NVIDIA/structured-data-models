import itertools
import random

import torch

from sdm import Stype, TableTensor
from sdm.processing import (
    AlignCategories,
    Choice,
    Clip,
    ClipSigma,
    DropConstantColumns,
    Identity,
    ImputeMean,
    PowerTransform,
    Processor,
    Recipe,
    ReduceEstimators,
    ShuffleCategories,
    ShuffleColumns,
    Softmax,
    Standardize,
    StypeDispatch,
    TaskDispatch,
    ToNumerical,
)
from sdm.tensor import EnsembleTable


class _TabICLv2EnsemblePlan:
    """Build the coupled member permutations required by TabICLv2."""

    def __init__(self) -> None:
        self._member_ids: tuple[int, ...] = ()
        self._num_members = 0
        self._seed = 0
        self._classes: tuple[object, ...] | None = None
        self._canonical_columns: tuple[str, ...] | None = None
        self._num_features: int | None = None
        self._feature_permutations: tuple[tuple[int, ...], ...] = ()
        self._class_permutations: tuple[tuple[int, ...] | None, ...] = ()

    def initialize(
        self,
        target: TableTensor,
        member_ids: tuple[int, ...],
        num_members: int,
        seed: int,
    ) -> None:
        self._member_ids = member_ids
        self._num_members = num_members
        self._seed = seed
        self._classes = (
            tuple(
                AlignCategories(sort_by="value")
                .fit_transform(target)
                .categorical.categories[0]
                .tolist()
            )
            if target.categorical.size(-1) == 1
            else None
        )
        self._canonical_columns = (
            tuple(str(value) for value in self._classes)
            if self._classes is not None
            else None
        )
        self._num_features = None

    @staticmethod
    def _latin_permutations(
        num_features: int,
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

        rows = square(list(range(num_features)))
        rng.shuffle(rows)
        columns = list(zip(*rows))
        rng.shuffle(columns)
        return [list(permutation) for permutation in columns]

    def _build(self, num_features: int) -> None:
        if self._num_features == num_features:
            return
        if self._num_members == 1:
            features = [list(range(num_features))]
        elif num_features <= 4000:
            features = self._latin_permutations(num_features, self._seed)
        else:
            rng = random.Random(self._seed)
            indices = list(range(num_features))
            features = [
                rng.sample(indices, num_features)
                for _ in range(self._num_members)
            ]

        if self._classes is None:
            classes: list[list[int] | None] = [None]
        else:
            indices = list(range(len(self._classes)))
            classes = (
                [indices]
                if self._num_members == 1
                else [
                    indices[-offset:] + indices[:-offset]
                    for offset in range(len(indices))
                ]
            )

        pairs = list(itertools.product(features, classes))
        random.Random(self._seed).shuffle(pairs)
        members = [pair for pair in pairs for _ in range(2)][
            : self._num_members
        ]
        self._num_features = num_features
        self._feature_permutations = tuple(
            tuple(feature) for feature, _ in members
        )
        self._class_permutations = tuple(
            None if values is None else tuple(values) for _, values in members
        )

    def option_ids(self, num_members: int) -> tuple[int, ...]:
        member_ids = (
            self._member_ids
            if len(self._member_ids) == num_members
            else tuple(range(num_members))
        )
        return tuple(member_id % 2 for member_id in member_ids)

    def column_permutations(
        self,
        member_ids: tuple[int, ...],
        num_columns: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        if self._num_members == 0:
            return tuple(
                tuple(range(num_columns[position]))
                for position in range(len(member_ids))
            )
        self._build(num_columns[0])
        active_member_ids = (
            self._member_ids
            if len(self._member_ids) == len(member_ids)
            else member_ids
        )
        return tuple(
            self._feature_permutations[member_id]
            for member_id in active_member_ids
        )

    def category_permutations(
        self,
        member_ids: tuple[int, ...],
        category_counts: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        if self._num_members == 0 or self._num_features is None:
            return tuple(
                tuple(tuple(range(count)) for count in counts)
                for counts in category_counts
            )
        active_member_ids = (
            self._member_ids
            if len(self._member_ids) == len(member_ids)
            else member_ids
        )
        permutations = []
        for member_id in active_member_ids:
            permutation = self._class_permutations[member_id]
            assert permutation is not None
            permutations.append((permutation,))
        return tuple(permutations)


class _TabICLv2Choice(Choice):
    """Select normalization from the global TabICLv2 member index."""

    def __init__(
        self,
        *args: object,
        plan: _TabICLv2EnsemblePlan,
    ) -> None:
        super().__init__(*args, selection="round_robin")
        self._ensemble_plan = plan

    def _draw_option_ids(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[int, ...]:
        del generator
        return self._ensemble_plan.option_ids(ensemble_table.num_members)


class _CanonicalClassOutput(Processor):
    """Reorder TabICLv2 class outputs to the fitted canonical space."""

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, plan: _TabICLv2EnsemblePlan) -> None:
        super().__init__()
        self._ensemble_plan = plan

    def _transform(self, table: TableTensor) -> TableTensor:
        canonical = self._ensemble_plan._canonical_columns
        if canonical is None:
            return table

        columns = table.columns[Stype.numerical]
        if columns == canonical:
            return table
        indices = tuple(columns.index(column) for column in canonical)
        numerical = (
            table.numerical.narrow(-1, indices[0], 1)
            if len(indices) == 1
            else torch.cat(
                tuple(
                    table.numerical.narrow(-1, index, 1) for index in indices
                ),
                dim=-1,
            )
        )
        return table.__class__(
            columns={Stype.numerical: canonical},
            numerical=numerical,
        )


class _TabICLv2Recipe(Recipe):
    """Recipe that prepares a coupled TabICLv2 ensemble plan."""

    _ensemble_plan: _TabICLv2EnsemblePlan

    def _prepare_members(
        self,
        *,
        y_context: TableTensor,
        member_ids: tuple[int, ...],
        num_members_total: int,
        generator: torch.Generator | None,
    ) -> None:
        self._ensemble_plan.initialize(
            y_context,
            member_ids,
            num_members_total,
            (
                generator.initial_seed()
                if generator is not None
                else torch.initial_seed()
            ),
        )


def default_recipe() -> Recipe:  # noqa: D103
    plan = _TabICLv2EnsemblePlan()
    recipe = _TabICLv2Recipe(
        features=[
            StypeDispatch(
                categorical=[
                    AlignCategories(sort_by="value"),
                    ToNumerical(),
                ],
            ),
            StypeDispatch(
                numerical=[
                    ImputeMean(),
                    DropConstantColumns(),
                    Standardize(epsilon=1e-6),
                    Clip(min_value=-100.0, max_value=100.0),
                    _TabICLv2Choice(
                        Identity(),
                        PowerTransform(),
                        plan=plan,
                    ),
                    ClipSigma(threshold=4.0),
                    ShuffleColumns(
                        method="shift",
                        _ensemble_permutations=plan.column_permutations,
                    ),
                ],
            ),
        ],
        target=[
            StypeDispatch(
                categorical=[
                    AlignCategories(sort_by="value"),
                    ShuffleCategories(
                        method="shift",
                        _ensemble_permutations=plan.category_permutations,
                    ),
                ],
                numerical=Standardize(),
            ),
        ],
        output=[
            ReduceEstimators(method="mean"),
            TaskDispatch(
                classification=Softmax(temperature=0.9),
            ),
            _CanonicalClassOutput(plan),
        ],
    )
    recipe._ensemble_plan = plan
    return recipe
