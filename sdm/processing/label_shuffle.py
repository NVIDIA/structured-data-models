from typing import Literal

import torch
from torch import Tensor

from sdm.processing.base import InvertibleMixin, Processor

LabelShuffleMethod = Literal["shift", "random", "none"]


class LabelShuffle(Processor, InvertibleMixin):
    """Permute integer-encoded target labels for ensemble diversity.

    The unresolved processor represents the single-estimator view and is an
    identity transform. Use :meth:`resolve` with ``estimator > 0`` to obtain a
    concrete non-identity view.

    Args:
        method: Permutation strategy. ``"none"`` disables permutation,
            ``"shift"`` uses deterministic cyclic shifts, and ``"random"``
            uses a deterministic seed from ``generator``.
        n_classes: Optional number of target classes. If omitted, ``fit``
            learns it from the non-negative labels in the input.
        generator: Optional torch generator whose initial seed drives
            ``"random"`` permutations without advancing generator state.
    """

    def __init__(
        self,
        method: LabelShuffleMethod = "shift",
        *,
        n_classes: int | None = None,
        generator: torch.Generator | None = None,
        _estimator: int = 0,
    ) -> None:
        super().__init__()
        if method not in {"shift", "random", "none"}:
            raise ValueError(
                "method must be one of 'shift', 'random', or 'none'"
            )
        if n_classes is not None and n_classes < 0:
            raise ValueError("n_classes must be non-negative")
        if _estimator < 0:
            raise ValueError("estimator must be non-negative")
        self.method = method
        self.n_classes = n_classes
        self.generator = generator
        self._estimator = _estimator
        self._fitted = n_classes is not None

    def _fit(self, input: Tensor) -> None:
        labels = _valid_labels(input)
        inferred = 0 if labels.numel() == 0 else int(labels.max().item()) + 1

        if self.n_classes is not None:
            if inferred > self.n_classes:
                raise ValueError(
                    "Expected label indices to be less than n_classes "
                    f"(got max label {inferred - 1} and "
                    f"n_classes={self.n_classes})"
                )
            return

        self.n_classes = inferred

    def resolve(
        self,
        *,
        estimator: int = 0,
        generator: torch.Generator | None = None,
    ) -> "LabelShuffle":
        """Return a concrete label permutation for one estimator view."""
        return self.__class__(
            method=self.method,
            n_classes=self.n_classes,
            generator=self.generator if generator is None else generator,
            _estimator=estimator,
        )

    def forward(self, input: Tensor) -> Tensor:
        """Map original label ids into the estimator-specific label space."""
        return self._map_labels(input, inverse=False)

    def _inverse_transform(self, input: Tensor) -> Tensor:
        return self._map_labels(input, inverse=True)

    def correct_output(self, input: Tensor) -> Tensor:
        """Map class scores back to the original class order.

        This mirrors TabICL's ensemble aggregation correction: labels are
        transformed with ``permutation[label]``, while class-score outputs are
        corrected with ``output[..., permutation]`` before averaging.
        """
        self._check_is_fitted()
        permutation = self._permutation(input.device)
        if input.size(-1) != permutation.numel():
            raise ValueError(
                "Expected the output class dimension to match n_classes "
                f"(got {input.size(-1)} and {permutation.numel()})"
            )
        if _is_identity(permutation):
            return input
        return input.index_select(-1, permutation)

    def _map_labels(self, input: Tensor, *, inverse: bool) -> Tensor:
        permutation = self._permutation(input.device)
        valid = _valid_mask(input)
        if not valid.any():
            return input

        labels = _valid_labels(input)
        if labels.max() >= permutation.numel():
            raise ValueError(
                "Expected label indices to be less than n_classes "
                f"(got max label {int(labels.max().item())} and "
                f"n_classes={permutation.numel()})"
            )

        if inverse:
            permutation = permutation.argsort()
        if _is_identity(permutation):
            return input

        output = input.clone()
        output[valid] = permutation[labels].to(input.dtype)
        return output

    def _permutation(self, device: torch.device) -> Tensor:
        n_classes = 0 if self.n_classes is None else self.n_classes
        if n_classes <= 1 or self._estimator == 0 or self.method == "none":
            return torch.arange(n_classes, device=device)

        if self.method == "shift":
            offset = self._estimator % n_classes
            return (
                torch.arange(n_classes, device=device) - offset
            ) % n_classes

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._random_seed(n_classes))
        return torch.randperm(n_classes, generator=generator).to(device=device)

    def _random_seed(self, n_classes: int) -> int:
        base_seed = (
            0 if self.generator is None else self.generator.initial_seed()
        )
        return (base_seed + self._estimator * 1_000_003 + n_classes) % (
            2**63 - 1
        )


def _valid_mask(input: Tensor) -> Tensor:
    valid = input >= 0
    if torch.is_floating_point(input):
        valid = valid & ~torch.isnan(input)
    return valid


def _valid_labels(input: Tensor) -> Tensor:
    labels = input[_valid_mask(input)]
    if torch.is_floating_point(labels) and not torch.equal(
        labels, labels.round()
    ):
        raise ValueError("Expected label indices to be integer-valued")
    return labels.to(torch.long)


def _is_identity(permutation: Tensor) -> bool:
    return bool(
        torch.equal(
            permutation,
            torch.arange(permutation.numel(), device=permutation.device),
        )
    )
