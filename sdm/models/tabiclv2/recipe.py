import itertools
import random

from sdm import TableTensor
from sdm.processing import (
    AlignCategories,
    Choice,
    Clip,
    ClipSigma,
    DropConstantColumns,
    Identity,
    ImputeMean,
    PowerTransform,
    Recipe,
    ReduceEstimators,
    ShuffleCategories,
    ShuffleColumns,
    Softmax,
    Standardize,
    StypeDispatch,
    TargetDecode,
    TaskDispatch,
    ToNumerical,
)


class _TabICLv2EnsemblePlan:
    """Build the coupled member permutations required by TabICLv2."""

    def __init__(self) -> None:
        self._num_members = 0
        self._seed = 0
        self._classes: tuple[object, ...] | None = None
        self._num_features: int | None = None
        self._feature_permutations: tuple[tuple[int, ...], ...] = ()
        self._class_permutations: tuple[tuple[int, ...] | None, ...] = ()

    def initialize(
        self,
        target: TableTensor,
        num_members: int,
        seed: int,
    ) -> None:
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

    def column_permutations(
        self,
        member_ids: tuple[int, ...],
        num_columns: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        self._build(num_columns[0])
        return tuple(
            self._feature_permutations[member] for member in member_ids
        )

    def category_permutations(
        self,
        member_ids: tuple[int, ...],
        category_counts: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        del category_counts
        assert self._num_features is not None
        permutations = []
        for member in member_ids:
            permutation = self._class_permutations[member]
            assert permutation is not None
            permutations.append((permutation,))
        return tuple(permutations)


def default_recipe(*, reference_ensemble: bool = True) -> Recipe:  # noqa: D103
    plan = _TabICLv2EnsemblePlan() if reference_ensemble else None
    recipe = Recipe(
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
                    Choice(
                        Identity(),
                        PowerTransform(),
                        selection=(
                            "round_robin" if reference_ensemble else "random"
                        ),
                    ),
                    ClipSigma(threshold=4.0),
                    ShuffleColumns(
                        method="latin" if reference_ensemble else "shift",
                        _ensemble_permutations=(
                            plan.column_permutations
                            if plan is not None
                            else None
                        ),
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
                        _ensemble_permutations=(
                            plan.category_permutations
                            if plan is not None
                            else None
                        ),
                    ),
                ],
                numerical=Standardize(),
            ),
        ],
        output=[
            TargetDecode(),
            ReduceEstimators(method="mean"),
            TaskDispatch(
                classification=Softmax(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
    recipe._ensemble_plan = plan
    return recipe
