from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor

LabelShuffleMethod = Literal["shift", "random", "none"]


class LabelShuffle(Processor, InvertibleMixin):
    """Apply the `"TabICL" <https://arxiv.org/abs/2502.05564>`_ label view.

    The view is drawn once, at construction, and consumes a single draw
    from the global CPU generator; seed with :func:`torch.manual_seed` to
    make the view reproducible. Constructing the same recipe repeatedly
    therefore yields ensemble members with independently drawn label
    mappings, mirroring :class:`~sdm.processing.FeaturePermute`.

    Only numerical target columns are supported.
    Negative labels and NaN values are preserved unchanged.

    Args:
        method: Permutation strategy. ``"none"`` disables permutation,
            ``"shift"`` cyclically shifts the labels by a drawn offset, and
            ``"random"`` remaps the labels with a drawn permutation.
        n_classes: Optional number of target classes. If omitted, ``fit``
            learns it from the non-negative labels in the input.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        method: LabelShuffleMethod = "shift",
        *,
        n_classes: int | None = None,
    ) -> None:
        super().__init__()
        if method not in {"shift", "random", "none"}:
            raise ValueError(
                "method must be one of 'shift', 'random', or 'none'"
            )
        if n_classes is not None and n_classes < 0:
            raise ValueError("n_classes must be non-negative")
        self.method = method
        self.n_classes = n_classes
        self._draw = int(torch.randint(2**63 - 1, (1,)).item())
        self._fitted = n_classes is not None

    def _fit(self, input: TableTensor) -> None:
        labels = _valid_labels(input.numerical)
        # Class count is fitted Python metadata, so one sync is required.
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

    def _transform(self, input: TableTensor) -> TableTensor:
        """Map original label ids into the drawn label space."""
        numerical = self._map_labels(input.numerical, inverse=False)
        if numerical is input.numerical:
            return input
        return input.replace_blocks(numerical=numerical)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        numerical = self._map_labels(input.numerical, inverse=True)
        if numerical is input.numerical:
            return input
        return input.replace_blocks(numerical=numerical)

    def correct_output(self, input: Tensor) -> Tensor:
        """Map class scores back to the original class order.

        This mirrors TabICL's ensemble aggregation correction: labels are
        transformed with ``permutation[label]``, while class-score outputs are
        corrected with ``output[..., permutation]`` before averaging.

        Args:
            input: Class scores with shape ``[..., C]``, where ``C`` is the
                number of classes.

        Returns:
            Class scores with shape ``[..., C]`` in the original class order.
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
        if n_classes <= 1 or self.method == "none":
            return torch.arange(n_classes, device=device)

        if self.method == "shift":
            offset = self._draw % n_classes
            return (
                torch.arange(n_classes, device=device) - offset
            ) % n_classes

        # A CPU generator makes drawn views identical across CPU and CUDA.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._draw)
        return torch.randperm(n_classes, generator=generator).to(device=device)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )


def _valid_mask(input: Tensor) -> Tensor:
    valid = input >= 0
    if input.is_floating_point():
        valid = valid & ~input.isnan()
    return valid


def _valid_labels(input: Tensor) -> Tensor:
    labels = input[_valid_mask(input)]
    if labels.is_floating_point() and not labels.equal(labels.round()):
        raise ValueError("Expected label indices to be integer-valued")
    return labels.to(torch.long)


def _is_identity(permutation: Tensor) -> bool:
    return permutation.equal(
        torch.arange(permutation.numel(), device=permutation.device)
    )
