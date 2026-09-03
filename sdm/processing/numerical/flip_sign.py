from typing import cast

import torch

from sdm import Stype
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.tensor import EnsembleTable


class FlipSign(EnsembleProcessor, EnsembleInvertibleMixin):
    """Randomly negate numerical feature columns.

    A sign is sampled independently for each numerical column when the
    processor is fitted and reused for every row and subsequent transform.
    Each ensemble member receives independent signs.

    Args:
        probability: Probability of negating each numerical column.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")
        self.probability = probability
        self.register_buffer("_signs", torch.empty(0, dtype=torch.int8))
        self._sign_slices: tuple[tuple[int, int], ...] = ()

    def get_extra_state(self) -> tuple[tuple[int, int], ...]:
        r""":meta private:"""  # noqa: D415
        return self._sign_slices

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._sign_slices = cast(tuple[tuple[int, int], ...], state)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        groups = tuple(ensemble_table)
        sign_slices = []
        num_signs = 0
        for group_id, _ in ensemble_table._locations:
            width = groups[group_id].numerical.size(-1)
            sign_slices.append((num_signs, width))
            num_signs += width

        signs = torch.empty(
            num_signs,
            dtype=torch.int8,
            device=ensemble_table.device,
        )
        if self.probability == 0.0:
            signs.fill_(1)
        elif self.probability == 1.0:
            signs.fill_(-1)
        else:
            signs.bernoulli_(self.probability, generator=generator)
            signs.mul_(-2).add_(1)
        self._signs = signs
        self._sign_slices = tuple(sign_slices)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._sign_slices) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._sign_slices)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )

        sign_slices_by_group: list[list[tuple[int, int]]] = [
            [] for _ in range(ensemble_table.num_groups)
        ]
        locations = []
        next_position = [0] * ensemble_table.num_groups
        for member_id, (group_id, _) in enumerate(ensemble_table._locations):
            sign_slices_by_group[group_id].append(self._sign_slices[member_id])
            locations.append((group_id, next_position[group_id]))
            next_position[group_id] += 1

        groups = []
        for group_id, sign_slices in enumerate(sign_slices_by_group):
            group = ensemble_table.expanded_group(group_id)
            expected_width = group.numerical.size(-1)
            if any(width != expected_width for _, width in sign_slices):
                raise RuntimeError(
                    f"{self.__class__.__name__!r} was fitted for a different "
                    "numerical schema."
                )

            sign_start = sign_slices[0][0]
            if all(
                sign_slice
                == (sign_start + position * expected_width, expected_width)
                for position, sign_slice in enumerate(sign_slices)
            ):
                sign = self._signs.narrow(
                    0,
                    sign_start,
                    len(sign_slices) * expected_width,
                ).view(len(sign_slices), expected_width)
            else:
                sign = torch.stack(
                    [
                        self._signs.narrow(0, slice_start, width)
                        for slice_start, width in sign_slices
                    ]
                )
            sign = sign.to(dtype=group.numerical.dtype)
            sign = sign.view(
                sign.size(0),
                *([1] * (group.numerical.dim() - 2)),
                sign.size(1),
            )
            groups.append(
                group.replace_blocks(numerical=group.numerical * sign)
            )

        return EnsembleTable._from_groups(groups, locations)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._transform_ensemble(ensemble_table)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"probability={self.probability!r})"
        )
