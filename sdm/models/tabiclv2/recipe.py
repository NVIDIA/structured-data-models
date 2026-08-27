import random

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import Stype
from sdm.tensor import EnsembleTable


class _TabICLv2EstimatorPlan:
    """Build the compact Latin state required for TabICLv2 parity."""

    def __init__(self) -> None:
        self._num_classes = 1
        self._num_members = 1
        self._seed = 0

    def latin_state(
        self,
        n_features: int,
        member_ids: tuple[int, ...],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if n_features == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            patterns = torch.empty(
                len(member_ids), dtype=torch.long, device=device
            )
            return empty, empty, patterns
        if self._num_members == 1:
            base = torch.arange(n_features, device=device)
            rows = -base
            patterns = torch.zeros(1, dtype=torch.long, device=device)
            return base, rows, patterns

        capacity = 2 * n_features * self._num_classes
        if self._num_members > capacity:
            raise ValueError(
                "TabICLv2's exact estimator plan supports at most "
                f"{capacity} members for {n_features} features and "
                f"{self._num_classes} classes"
            )

        base_values, row_values, pattern_order = self._draw_latin(
            n_features,
            self._seed,
        )
        pair_ids = list(range(n_features * self._num_classes))
        random.Random(self._seed).shuffle(pair_ids)
        return (
            torch.tensor(base_values, device=device),
            torch.tensor(row_values, device=device),
            torch.tensor(
                [
                    pattern_order[
                        pair_ids[member_id // 2] // self._num_classes
                    ]
                    for member_id in member_ids
                ],
                device=device,
            ),
        )

    @staticmethod
    def _draw_latin(
        n_features: int,
        seed: int,
    ) -> tuple[list[int], list[int], list[int]]:
        rng = random.Random(seed)
        tree = [0] + [index & -index for index in range(1, n_features + 1)]

        def pop(order: int) -> int:
            index = 0
            step = 1 << (n_features.bit_length() - 1)
            while step:
                candidate = index + step
                if candidate <= n_features and tree[candidate] <= order:
                    index = candidate
                    order -= tree[candidate]
                step >>= 1
            position = index + 1
            while position <= n_features:
                tree[position] -= 1
                position += position & -position
            return index

        base = [
            pop(rng.randrange(remaining))
            for remaining in range(n_features, 1, -1)
        ]
        base.append(pop(0))
        rows = list(range(n_features))
        rng.shuffle(rows)
        patterns = list(range(n_features))
        rng.shuffle(patterns)
        return base, rows, patterns


class _InitializeTabICLv2EstimatorPlan(
    sp.EnsembleProcessor,
    sp.EnsembleInvertibleMixin,
):
    """Initialize the estimator plan before target preprocessing."""

    handles_stypes = frozenset(Stype)
    requires_fit = True

    def __init__(self, plan: _TabICLv2EstimatorPlan) -> None:
        super().__init__()
        self._plan = plan

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._plan._num_members = ensemble_table.num_members
        self._plan._seed = (
            generator.initial_seed()
            if generator is not None
            else torch.initial_seed()
        )
        target = ensemble_table.table(0)
        if target.categorical.size(-1) == 0:
            self._plan._num_classes = 1
            return
        codes = target.categorical.code[..., 0]
        self._plan._num_classes = codes[codes >= 0].unique().numel()

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table


class _TabICLv2ShuffleColumns(sp.ShuffleColumns):
    """Apply the exact compact TabICLv2 Latin estimator plan."""

    def __init__(self, plan: _TabICLv2EstimatorPlan) -> None:
        super().__init__(method="latin")
        self._plan = plan

    def _latin_states(
        self,
        schemas: tuple[tuple[str, ...], ...],
        members: tuple[tuple[int, ...], ...],
        devices: tuple[torch.device, ...],
        *,
        generator: torch.Generator | None,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        del generator
        return tuple(
            self._plan.latin_state(len(schema), member_ids, device)
            for schema, member_ids, device in zip(
                schemas, members, devices, strict=True
            )
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}ShuffleColumns(method='latin')"


def default_recipe() -> sp.Recipe:  # noqa: D103
    plan = _TabICLv2EstimatorPlan()
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.ToNumerical(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.ImputeMean(),
                    sp.DropConstantColumns(),
                    sp.Standardize(epsilon=1e-6),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.Choice(
                        sp.Identity(),
                        sp.PowerTransform(),
                        method="round_robin",
                    ),
                    sp.ClipSigma(threshold=4.0),
                    _TabICLv2ShuffleColumns(plan),
                ],
            ),
        ],
        target=[
            _InitializeTabICLv2EstimatorPlan(plan),
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=sp.Standardize(),
            ),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(
                classification=sp.Softmax(temperature=0.9),
            ),
        ],
    )
