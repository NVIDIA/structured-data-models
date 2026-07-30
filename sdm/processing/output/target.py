from collections.abc import Sequence

import torch

from sdm.processing.base import Processor
from sdm.processing.ensemble import (
    EnsembleProcessor,
    _stack_physical,
)
from sdm.stype import Stype
from sdm.tensor import TableTensor


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
        self._class_indices: tuple[torch.Tensor, ...] = ()
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
        self._class_indices = tuple(class_indices)
        self._num_members = num_members

    def _transform(self, table: TableTensor) -> TableTensor:
        if self.target is None:
            raise RuntimeError("TargetDecode is not bound to a fitted Recipe.")
        if table.size(0) != self._num_members:
            raise ValueError(
                "Expected one model output per fitted ensemble member."
            )

        outputs = tuple(table[member] for member in range(self._num_members))
        if self._canonical_classes is None:
            return _stack_physical(
                self.target.inverse_transform_members(outputs)
            )

        columns = tuple(str(value) for value in self._canonical_classes)
        decoded: list[TableTensor] = []
        for output, indices in zip(outputs, self._class_indices):
            if output.numerical.size(-1) != indices.numel():
                raise ValueError(
                    "Model output width does not match the fitted class count."
                )
            decoded.append(
                TableTensor(
                    columns={Stype.numerical: columns},
                    numerical=output.numerical.index_select(-1, indices),
                )
            )
        return _stack_physical(decoded)
