from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class FeaturePermute(Processor, InvertibleMixin):
    """Permute the numerical feature columns.

    The permutation is drawn from the global CPU generator when the
    processor is fitted; seed with :func:`torch.manual_seed` to make it
    reproducible. Convert non-numerical feature stypes before this step,
    for example with :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"none"`` disables permutation,
            ``"shift"`` cyclically shifts the columns by a drawn offset, and
            ``"random"`` permutes the columns with a drawn permutation.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        method: Literal["shift", "random", "none"] = "shift",
    ) -> None:
        super().__init__()
        self.method = method
        self.register_buffer(
            "permutation",
            torch.empty(0, dtype=torch.long),
        )

    def _fit(self, input: TableTensor) -> None:
        n_features = input.numerical.size(-1)
        device = input.numerical.device
        if self.method == "none" or n_features <= 1:
            self.permutation = torch.arange(n_features, device=device)
        elif self.method == "shift":
            offset = int(torch.randint(n_features, (1,)).item())
            self.permutation = (
                torch.arange(n_features, device=device) + offset
            ) % n_features
        else:
            # The global CPU generator makes drawn permutations identical
            # across CPU and CUDA.
            self.permutation = torch.randperm(n_features).to(device=device)

    def _transform(self, input: TableTensor) -> TableTensor:
        """Reorder the numerical block with the fitted permutation."""
        if self.method == "none":
            return input
        return self._permute(input, self.permutation)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        if self.method == "none":
            return input
        return self._permute(input, self.permutation.argsort())

    def _permute(
        self,
        input: TableTensor,
        permutation: Tensor,
    ) -> TableTensor:
        # Column names are Python metadata, so mapping them requires one sync.
        indices = permutation.tolist()
        return input.__class__(
            columns={
                Stype.numerical.value: tuple(
                    input.columns[Stype.numerical][index] for index in indices
                )
            },
            numerical=input.numerical.index_select(-1, permutation),
        )
