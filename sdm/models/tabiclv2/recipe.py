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
    """Reproduce TabICL's paired normalization and permutation plan."""

    def __init__(self) -> None:
        self._num_members = 0
        self._seed = 0
        self._canonical_classes: tuple[object, ...] | None = None
        self._num_features: int | None = None
        self._feature_permutations: tuple[tuple[int, ...], ...] = ()
        self._class_permutations: tuple[tuple[int, ...] | None, ...] = ()

    def initialize(
        self,
        *,
        target: TableTensor,
        num_members: int,
        seed: int,
    ) -> None:
        self._num_members = num_members
        self._seed = seed
        if target.categorical.size(-1) == 1:
            aligned = AlignCategories(sort_by="value").fit_transform(target)
            self._canonical_classes = tuple(
                aligned.categorical.categories[0].tolist()
            )
        else:
            self._canonical_classes = None
        self._num_features = None
        self._feature_permutations = ()
        self._class_permutations = ()

    def canonical_classes(self) -> tuple[object, ...] | None:
        return self._canonical_classes

    @staticmethod
    def _latin_permutations(
        num_features: int,
        *,
        seed: int,
    ) -> list[list[int]]:
        rng = random.Random(seed)

        def reduced_latin_square(symbols: list[int]) -> list[list[int]]:
            if len(symbols) == 1:
                return [symbols]
            symbol = rng.choice(symbols)
            symbols.remove(symbol)
            square = reduced_latin_square(symbols)
            square.append(square[0].copy())
            for index in range(len(square)):
                square[index].insert(index, symbol)
            return square

        square = reduced_latin_square(list(range(num_features)))
        rng.shuffle(square)
        transposed = list(zip(*square))
        rng.shuffle(transposed)
        return [list(permutation) for permutation in transposed]

    def _build(self, num_features: int) -> None:
        if self._num_features == num_features:
            return
        if num_features < 1:
            raise ValueError(
                "TabICLv2 requires at least one non-constant feature."
            )

        if self._num_members == 1:
            feature_patterns = [list(range(num_features))]
        elif num_features <= 4000:
            feature_patterns = self._latin_permutations(
                num_features,
                seed=self._seed,
            )
        else:
            rng = random.Random(self._seed)
            indices = list(range(num_features))
            feature_patterns = [
                rng.sample(indices, num_features)
                for _ in range(self._num_members)
            ]

        if self._canonical_classes is None:
            class_patterns: list[list[int] | None] = [None]
        else:
            num_classes = len(self._canonical_classes)
            if self._num_members == 1:
                class_patterns = [list(range(num_classes))]
            else:
                indices = list(range(num_classes))
                class_patterns = [
                    indices[-offset:] + indices[:-offset]
                    for offset in range(num_classes)
                ]

        paired = list(itertools.product(feature_patterns, class_patterns))
        random.Random(self._seed).shuffle(paired)
        member_pairs = [pair for pair in paired for _ in ("none", "power")][
            : self._num_members
        ]
        if len(member_pairs) != self._num_members:
            raise ValueError(
                "The requested TabICLv2 ensemble has more members than "
                "the reference permutation plan can produce."
            )

        self._num_features = num_features
        self._feature_permutations = tuple(
            tuple(feature) for feature, _ in member_pairs
        )
        self._class_permutations = tuple(
            None if classes is None else tuple(classes)
            for _, classes in member_pairs
        )

    def column_permutations(
        self,
        *,
        member_ids: tuple[int, ...],
        num_columns: tuple[int, ...],
        table_scope: str,
    ) -> tuple[tuple[int, ...], ...] | None:
        if table_scope != "features":
            return None
        if len(set(num_columns)) != 1:
            raise ValueError(
                "TabICLv2's reference plan requires one feature width."
            )
        self._build(num_columns[0])
        return tuple(
            self._feature_permutations[member] for member in member_ids
        )

    def category_permutations(
        self,
        *,
        member_ids: tuple[int, ...],
        category_counts: tuple[tuple[int, ...], ...],
        table_scope: str,
    ) -> tuple[tuple[tuple[int, ...], ...], ...] | None:
        if table_scope != "target" or self._canonical_classes is None:
            return None
        num_classes = len(self._canonical_classes)
        if any(counts != (num_classes,) for counts in category_counts):
            raise ValueError(
                "TabICLv2's reference plan requires one target category."
            )
        if self._num_features is None:
            raise RuntimeError(
                "Feature planning must run before target permutation planning."
            )
        permutations = []
        for member in member_ids:
            permutation = self._class_permutations[member]
            assert permutation is not None
            permutations.append((permutation,))
        return tuple(permutations)


def default_recipe(*, reference_ensemble: bool = True) -> Recipe:  # noqa: D103
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
                        method="latin" if reference_ensemble else "shift"
                    ),
                ],
            ),
        ],
        target=[
            StypeDispatch(
                categorical=[
                    AlignCategories(sort_by="value"),
                    ShuffleCategories(method="shift"),
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
    if reference_ensemble:
        recipe._ensemble_plan = _TabICLv2EnsemblePlan()
    return recipe
