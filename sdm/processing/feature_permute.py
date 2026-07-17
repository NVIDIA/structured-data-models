from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class FeaturePermute(Processor, InvertibleMixin):
    """Permute the numerical feature columns.

    The permutation is drawn when the processor is fitted; pass
    ``generator`` to ``fit()`` to make it reproducible. Convert
    non-numerical feature stypes before this step, for example with
    :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            columns by a drawn offset, and ``"random"`` permutes the columns
            with a drawn permutation.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        method: Literal["shift", "random"] = "shift",
    ) -> None:
        super().__init__()
        self.method = method
        self.register_buffer(
            "permutation",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer(
            "inverse_permutation",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._permutation_indices: tuple[int, ...] = ()
        self._inverse_permutation_indices: tuple[int, ...] = ()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        n_features = table.numerical.size(-1)
        device = table.numerical.device
        if n_features <= 1:
            self.permutation = torch.arange(n_features, device=device)
        elif self.method == "shift":
            offset = int(
                torch.randint(
                    n_features,
                    (1,),
                    generator=generator,
                    device=device,
                ).item()
            )
            self.permutation = (
                torch.arange(n_features, device=device) + offset
            ) % n_features
        else:
            self.permutation = torch.randperm(
                n_features,
                generator=generator,
                device=device,
            )
        self.inverse_permutation = self.permutation.argsort()
        self._permutation_indices = self._to_indices(self.permutation)
        self._inverse_permutation_indices = self._to_indices(
            self.inverse_permutation
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Reorder the numerical block with the fitted permutation."""
        return self._permute(
            table,
            self.permutation,
            self._index_tuple(inverse=False),
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        indices = self._index_tuple(inverse=True)
        return self._permute(
            table,
            self.inverse_permutation,
            indices,
        )

    def _index_tuple(self, *, inverse: bool) -> tuple[int, ...]:
        if (
            inverse
            and self.inverse_permutation.numel() != self.permutation.numel()
        ):
            self.inverse_permutation = self.permutation.argsort()
            self._inverse_permutation_indices = ()

        permutation = self.inverse_permutation if inverse else self.permutation
        indices = (
            self._inverse_permutation_indices
            if inverse
            else self._permutation_indices
        )
        if len(indices) == permutation.numel():
            return indices

        indices = self._to_indices(permutation)
        if inverse:
            self._inverse_permutation_indices = indices
        else:
            self._permutation_indices = indices
        return indices

    @staticmethod
    def _to_indices(permutation: Tensor) -> tuple[int, ...]:
        return tuple(int(index) for index in permutation.tolist())

    def _permute(
        self,
        table: TableTensor,
        permutation: Tensor,
        indices: tuple[int, ...],
    ) -> TableTensor:
        return table.__class__(
            columns={
                Stype.numerical.value: tuple(
                    table.columns[Stype.numerical][index] for index in indices
                )
            },
            numerical=table.numerical.index_select(-1, permutation),
        )
