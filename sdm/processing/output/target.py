from collections.abc import Sequence

import torch

from sdm.processing.base import Processor
from sdm.processing.ensemble import EnsembleProcessor
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor


class TargetDecode(Processor):
    r"""Map member outputs back to their common fitted target space.

    A :class:`~sdm.processing.Recipe` binds the fitted target pipeline when it
    is fitted. Regression values are inverse-transformed per member, while
    classification logits are aligned to the common class order.
    """

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self) -> None:
        super().__init__()
        self.target: EnsembleProcessor | None = None
        self._canonical_classes: tuple[object, ...] | None = None
        self._class_indices: torch.Tensor | None = None
        self._num_members = 0

    def _bind(
        self,
        *,
        target: EnsembleProcessor,
        canonical_classes: tuple[object, ...] | None,
        class_indices: Sequence[torch.Tensor],
        num_members: int,
    ) -> None:
        self.target = target
        self._canonical_classes = canonical_classes
        self._class_indices = (
            torch.stack(tuple(class_indices))
            if canonical_classes is not None
            else None
        )
        self._num_members = num_members

    def _transform(self, table: TableTensor) -> TableTensor:
        if self.target is None:
            raise RuntimeError("TargetDecode is not bound to a fitted Recipe.")

        if self._canonical_classes is None:
            encoded = EnsembleTable._from_packed_representations(
                packed_representations=(table,),
                member_locations=tuple(
                    (0, member) for member in range(self._num_members)
                ),
                member_ids=tuple(range(self._num_members)),
            )
            return self.target.inverse_transform_ensemble(
                encoded
            ).materialize()

        assert self._class_indices is not None
        if table.numerical.size(-1) != self._class_indices.size(-1):
            raise ValueError(
                "Model output width does not match the fitted class count."
            )
        indices = self._class_indices.view(
            self._num_members,
            *([1] * (table.numerical.dim() - 2)),
            self._class_indices.size(-1),
        ).expand_as(table.numerical)
        return TableTensor(
            columns={
                Stype.numerical: tuple(
                    str(value) for value in self._canonical_classes
                )
            },
            numerical=table.numerical.gather(-1, indices),
        )
