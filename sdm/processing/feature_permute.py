from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor

FeaturePermuteMethod = Literal["latin", "shift", "random", "none"]


class FeaturePermute(Processor, InvertibleMixin):
    """Permute numerical feature columns for ensemble diversity.

    The unresolved processor represents the single-estimator view and is an
    identity transform. Use :meth:`resolve` with ``estimator > 0`` to obtain a
    concrete non-identity view.

    Only numerical columns are supported. Convert other feature stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"none"`` disables permutation,
            ``"latin"`` and ``"shift"`` use deterministic cyclic shifts, and
            ``"random"`` uses a deterministic seed from ``generator``.
        generator: Optional torch generator whose initial seed drives
            ``"random"`` permutations without advancing generator state.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        method: FeaturePermuteMethod = "latin",
        *,
        generator: torch.Generator | None = None,
        _estimator: int = 0,
    ) -> None:
        super().__init__()
        if method not in {"latin", "shift", "random", "none"}:
            raise ValueError(
                "method must be one of 'latin', 'shift', 'random', or 'none'"
            )
        if _estimator < 0:
            raise ValueError("estimator must be non-negative")
        self.method = method
        self.generator = generator
        self._estimator = _estimator

    def resolve(
        self,
        *,
        estimator: int = 0,
        generator: torch.Generator | None = None,
    ) -> "FeaturePermute":
        """Return a concrete permutation for one estimator view.

        Args:
            estimator: Zero-based estimator index.
            generator: Optional generator that overrides the configured one.

        Returns:
            The resolved feature permutation.
        """
        return self.__class__(
            method=self.method,
            generator=self.generator if generator is None else generator,
            _estimator=estimator,
        )

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

        return input.__class__(
            columns={
                Stype.numerical.value: tuple(
                    input.columns[Stype.numerical][index]
                    for index in permutation.tolist()
                )
            },
            numerical=numerical.index_select(-1, permutation),
        )

    def _permutation(
        self,
        n_features: int,
        device: torch.device,
    ) -> Tensor:
        if n_features <= 1 or self._estimator == 0 or self.method == "none":
            return torch.arange(n_features, device=device)

        if self.method in {"latin", "shift"}:
            offset = self._estimator % n_features
            return (
                torch.arange(n_features, device=device) + offset
            ) % n_features

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._random_seed(n_features))
        return torch.randperm(n_features, generator=generator).to(
            device=device
        )

    def _random_seed(self, n_features: int) -> int:
        base_seed = (
            0 if self.generator is None else self.generator.initial_seed()
        )
        return (base_seed + self._estimator * 1_000_003 + n_features) % (
            2**63 - 1
        )


def _is_identity(permutation: Tensor) -> bool:
    return bool(
        torch.equal(
            permutation,
            torch.arange(permutation.numel(), device=permutation.device),
        )
    )
