from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor

FeaturePermuteMethod = Literal["shift", "random", "none"]


class FeaturePermute(Processor, InvertibleMixin):
    """Apply the `"TabICL" <https://arxiv.org/abs/2502.05564>`_ feature view.

    The view is drawn once, at construction, and consumes a single draw
    from the global CPU generator; seed with :func:`torch.manual_seed` to
    make the view reproducible. Constructing the same recipe repeatedly
    therefore yields ensemble members with independently drawn feature
    orders, mirroring the planned ``Choice`` processor.

    Only numerical columns are supported. Convert other feature stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"none"`` disables permutation,
            ``"shift"`` cyclically shifts the columns by a drawn offset, and
            ``"random"`` permutes the columns with a drawn permutation.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self, method: FeaturePermuteMethod = "shift") -> None:
        super().__init__()
        if method not in {"shift", "random", "none"}:
            raise ValueError(
                "method must be one of 'shift', 'random', or 'none'"
            )
        self.method = method
        self._draw = int(torch.randint(2**63 - 1, (1,)).item())

    def _transform(self, input: TableTensor) -> TableTensor:
        """Permute the numerical feature block."""
        return self._apply_permutation(input, inverse=False)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        return self._apply_permutation(input, inverse=True)

    def _apply_permutation(
        self, input: TableTensor, *, inverse: bool
    ) -> TableTensor:
        numerical = input.numerical
        permutation = self._permutation(numerical.size(-1), numerical.device)
        if inverse:
            permutation = permutation.argsort()
        if _is_identity(permutation):
            return input

        # Column names are Python metadata, so mapping them requires one sync.
        indices = permutation.tolist()
        return input.__class__(
            columns={
                Stype.numerical.value: tuple(
                    input.columns[Stype.numerical][index] for index in indices
                )
            },
            numerical=numerical.index_select(-1, permutation),
        )

    def _permutation(
        self,
        n_features: int,
        device: torch.device,
    ) -> Tensor:
        if n_features <= 1 or self.method == "none":
            return torch.arange(n_features, device=device)

        if self.method == "shift":
            offset = self._draw % n_features
            return (
                torch.arange(n_features, device=device) + offset
            ) % n_features

        # A CPU generator makes drawn views identical across CPU and CUDA.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._draw)
        return torch.randperm(n_features, generator=generator).to(
            device=device
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )


def _is_identity(permutation: Tensor) -> bool:
    return permutation.equal(
        torch.arange(permutation.numel(), device=permutation.device)
    )
