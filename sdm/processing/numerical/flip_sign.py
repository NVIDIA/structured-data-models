import torch
from torch import Tensor

from sdm import Stype
from sdm.nn._buffer import BufferList
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
        self._signs: BufferList[Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        signs = []
        for group_id, group in enumerate(ensemble_table):
            shape = (
                ensemble_table.num_members_in_group(group_id),
                group.numerical.size(-1),
            )
            sign = group.numerical.new_ones(shape)
            if self.probability == 1.0:
                sign.neg_()
            elif self.probability > 0.0:
                flip = (
                    torch.rand(
                        shape,
                        device=group.device,
                        generator=generator,
                    )
                    < self.probability
                )
                sign.masked_fill_(flip, -1)
            signs.append(sign)
        self._signs = BufferList(signs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._signs) != ensemble_table.num_groups:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted for a different "
                "ensemble layout."
            )

        groups = []
        for group_id, sign in enumerate(self._signs):
            group = ensemble_table.expanded_group(group_id)
            expected_shape = (group.size(0), group.numerical.size(-1))
            if sign.size() != expected_shape:
                raise RuntimeError(
                    f"{self.__class__.__name__!r} was fitted for a different "
                    "ensemble layout."
                )
            sign = sign.view(
                sign.size(0),
                *([1] * (group.numerical.dim() - 2)),
                sign.size(1),
            )
            groups.append(
                group.replace_blocks(numerical=group.numerical * sign)
            )

        locations = []
        next_position = [0] * ensemble_table.num_groups
        for group_id, _ in ensemble_table._locations:
            locations.append((group_id, next_position[group_id]))
            next_position[group_id] += 1
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
