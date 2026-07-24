"""Official TabICLv2 ensemble-policy control for the TabArena example.

This module mirrors only TabICL 2.0.1 ensemble scheduling. SDM intentionally
continues to own feature and target preprocessing.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import (
    CategoricalAlign,
    Clip,
    ConstantFilter,
    EnsembleReduce,
    FeaturePermute,
    Identity,
    MeanImpute,
    Power,
    Recipe,
    SigmaClip,
    SoftmaxTemperature,
    StandardScale,
    StypeDispatch,
    TaskDispatch,
    ToNumerical,
)
from sdm.processing.base import Processor

Normalization = Literal["none", "power"]


@dataclass(frozen=True)
class EnsembleMemberPlan:
    """One fully specified member of the official TabICLv2 policy."""

    normalization: Normalization
    feature_permutation: tuple[int, ...]
    class_permutation: tuple[int, ...] | None


@dataclass(frozen=True)
class OfficialV2EnsemblePlan:
    """The fixed eight-member TabICLv2 2.0.1 ensemble schedule."""

    members: tuple[EnsembleMemberPlan, ...]

    @classmethod
    def build(
        cls,
        *,
        feature_count: int,
        class_count: int | None,
        random_state: int = 42,
    ) -> OfficialV2EnsemblePlan:
        """Build the policy after SDM's common feature filtering.

        The published generator makes Latin feature patterns, circular class
        shifts, shuffles their Cartesian product with ``random_state``, pairs
        each base member with ``none`` and ``power``, and truncates to eight.
        It has no duplicate fallback; this control therefore fails if fewer
        than four base configurations are available.
        """
        if feature_count < 1:
            raise ValueError("Official policy requires at least one feature")
        if class_count is not None and class_count < 1:
            raise ValueError("Official policy requires at least one class")

        feature_patterns = _official_feature_patterns(
            n_elements=feature_count,
            random_state=random_state,
        )
        class_patterns: list[tuple[int, ...] | None]
        if class_count is None:
            class_patterns = [None]
        else:
            class_patterns = [
                tuple(range(class_count - shift, class_count))
                + tuple(range(class_count - shift))
                for shift in range(class_count)
            ]

        base_members = [
            (feature_pattern, class_pattern)
            for feature_pattern in feature_patterns
            for class_pattern in class_patterns
        ]
        random.Random(random_state).shuffle(base_members)
        if len(base_members) < 4:
            raise ValueError(
                "Official eight-member policy needs at least four distinct "
                "feature/class configurations after SDM filtering; got "
                f"{len(base_members)}."
            )

        members = tuple(
            EnsembleMemberPlan(
                normalization=normalization,
                feature_permutation=feature_pattern,
                class_permutation=class_pattern,
            )
            for feature_pattern, class_pattern in base_members
            for normalization in ("none", "power")
        )[:8]
        return cls(members=members)


def _official_feature_patterns(
    *,
    n_elements: int,
    random_state: int,
) -> list[tuple[int, ...]]:
    """Port TabICL's seeded ``Shuffler(method='latin')`` output exactly."""
    rng = random.Random(random_state)
    if n_elements > 4000:
        return [
            tuple(rng.sample(range(n_elements), n_elements)) for _ in range(8)
        ]

    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(original_limit, 100_000))
    try:

        def build_square(symbols: list[int]) -> list[list[int]]:
            if len(symbols) == 1:
                return [symbols]
            symbol = rng.choice(symbols)
            symbols.remove(symbol)
            square = build_square(symbols)
            square.append(square[0].copy())
            for index in range(len(square)):
                square[index].insert(index, symbol)
            return square

        square = build_square(list(range(n_elements)))
        rng.shuffle(square)
        patterns = list(zip(*square))
        rng.shuffle(patterns)
    finally:
        sys.setrecursionlimit(original_limit)
    return [tuple(pattern) for pattern in patterns]


class FixedFeaturePermute(FeaturePermute):
    """Permute numerical features by an explicit, deterministic ordering."""

    def __init__(self, permutation: Sequence[int]) -> None:
        Processor.__init__(self)
        self._requested_permutation = tuple(permutation)
        self.register_buffer("permutation", torch.empty(0, dtype=torch.long))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        del generator
        permutation = _validated_permutation(
            self._requested_permutation,
            size=table.numerical.size(-1),
            label="feature",
        )
        self.permutation = torch.tensor(
            permutation,
            dtype=torch.long,
            device=table.numerical.device,
        )


class FixedCategoryCodePermute(Processor):
    """Apply one categorical target-code permutation.

    Missing codes are retained.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(self, permutation: Sequence[int]) -> None:
        super().__init__()
        self._requested_permutation = tuple(permutation)
        self.register_buffer("permutation", torch.empty(0, dtype=torch.long))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        del generator
        if table.categorical.size(-1) != 1:
            raise ValueError(
                "Fixed categorical target permutation requires exactly one "
                "categorical column"
            )
        permutation = _validated_permutation(
            self._requested_permutation,
            size=table.categorical.categories[0].numel(),
            label="class",
        )
        self.permutation = torch.tensor(
            permutation,
            dtype=torch.long,
            device=table.categorical.device,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        data = table.categorical.as_tensor().clone()
        codes = data[..., 0]
        valid = codes >= 0
        remapped = self.permutation[codes.clamp_min(0).to(torch.long)]
        data[..., 0] = torch.where(valid, remapped.to(codes.dtype), codes)
        categorical = CategoricalTensor(
            data=data,
            categories=[
                table.categorical.categories[0][self.permutation.argsort()]
            ],
        )
        return table.replace_blocks(categorical=categorical)


def _validated_permutation(
    permutation: Sequence[int],
    *,
    size: int,
    label: str,
) -> tuple[int, ...]:
    values = tuple(permutation)
    if len(values) != size:
        raise ValueError(
            f"Expected {size} {label} permutation entries, got {len(values)}"
        )
    if any(not isinstance(value, int) for value in values):
        raise ValueError(
            f"{label.capitalize()} permutation entries must be integers"
        )
    if set(values) != set(range(size)):
        raise ValueError(
            f"{label.capitalize()} permutation must be a bijection over "
            f"[0, {size})"
        )
    return values


def official_policy_recipe(member: EnsembleMemberPlan) -> Recipe:
    """Create one deterministic SDM recipe for an official-policy member."""
    normalization: Processor
    normalization = Identity() if member.normalization == "none" else Power()
    target: list[Processor] = [CategoricalAlign()]
    if member.class_permutation is not None:
        target.append(FixedCategoryCodePermute(member.class_permutation))
    return Recipe(
        features=[
            StypeDispatch(
                numerical=Identity(),
                categorical=[CategoricalAlign(), ToNumerical()],
            ),
            StypeDispatch(
                numerical=[
                    MeanImpute(),
                    ConstantFilter(),
                    StandardScale(epsilon=1e-6),
                    Clip(min_value=-100.0, max_value=100.0),
                    normalization,
                    SigmaClip(threshold=4.0),
                    FixedFeaturePermute(member.feature_permutation),
                ],
            ),
        ],
        target=StypeDispatch(
            categorical=target,
            numerical=StandardScale(),
        ),
        output=[
            EnsembleReduce(method="mean"),
            TaskDispatch(
                classification=SoftmaxTemperature(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
