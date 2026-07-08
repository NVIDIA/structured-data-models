from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class ClassShuffle(Processor, InvertibleMixin):
    """Permute the integer class labels of a numerical target column.

    The class permutation is drawn from the global CPU generator when the
    processor is fitted; seed with :func:`torch.manual_seed` to make it
    reproducible. The number of classes is learned from the non-negative
    labels during ``fit``. Negative labels and NaN values are preserved
    unchanged.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            labels by a drawn offset, and ``"random"`` remaps the labels
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

    def _fit(self, input: TableTensor) -> None:
        labels = _valid_labels(input.numerical)
        device = input.numerical.device
        n_classes = 0 if labels.numel() == 0 else int(labels.max().item()) + 1
        if n_classes <= 1:
            self.permutation = torch.arange(n_classes, device=device)
        elif self.method == "shift":
            offset = int(torch.randint(n_classes, (1,)).item())
            self.permutation = (
                torch.arange(n_classes, device=device) - offset
            ) % n_classes
        else:
            self.permutation = torch.randperm(n_classes).to(device=device)

    def _transform(self, input: TableTensor) -> TableTensor:
        numerical = self._map_labels(input.numerical, self.permutation)
        if numerical is input.numerical:
            return input
        return input.replace_blocks(numerical=numerical)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        numerical = self._map_labels(
            input.numerical,
            self.permutation.argsort(),
        )
        if numerical is input.numerical:
            return input
        return input.replace_blocks(numerical=numerical)

    def correct_output(self, input: Tensor) -> Tensor:
        """Map class scores back to the original class order.

        Labels are transformed with ``permutation[label]``, so class-score
        outputs are corrected with ``output[..., permutation]`` before
        ensemble averaging.

        Args:
            input: Class scores with shape ``[..., C]``, where ``C`` is the
                number of classes.

        Returns:
            Class scores with shape ``[..., C]`` in the original class order.
        """
        self._check_is_fitted()
        if input.size(-1) != self.permutation.numel():
            raise ValueError(
                "Expected the output class dimension to match the fitted "
                f"class count (got {input.size(-1)} and "
                f"{self.permutation.numel()})"
            )
        return input.index_select(-1, self.permutation)

    def _map_labels(self, input: Tensor, permutation: Tensor) -> Tensor:
        valid = _valid_mask(input)
        if not valid.any():
            return input

        labels = _valid_labels(input)
        if labels.max() >= permutation.numel():
            raise ValueError(
                "Expected label indices to be less than the fitted class "
                f"count (got max label {int(labels.max().item())} and "
                f"{permutation.numel()})"
            )

        output = input.clone()
        output[valid] = permutation[labels].to(input.dtype)
        return output


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
