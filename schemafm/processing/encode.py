from __future__ import annotations

import torch
from torch import Tensor

from schemafm.processing.base import InvertibleMixin, Processor


def _nan_mask(input: Tensor) -> Tensor:
    if input.is_floating_point():
        return torch.isnan(input)
    return torch.zeros_like(input, dtype=torch.bool)


def _integer_code_mask(input: Tensor, codes: Tensor) -> Tensor:
    if input.is_floating_point():
        return input == codes.to(input.dtype)
    return torch.ones_like(codes, dtype=torch.bool)


def _unknown_output(
    shape: torch.Size,
    *,
    template: Tensor,
    unknown_value: int,
) -> Tensor:
    if template.is_floating_point():
        return template.new_full(shape, torch.nan)
    if template.dtype == torch.bool:
        return template.new_zeros(shape)
    return template.new_full(shape, unknown_value)


class LabelEncode(Processor, InvertibleMixin):
    """Encode known 1D target labels as ordinal class indices."""

    def __init__(self, *, unknown_value: int = -1) -> None:
        super().__init__()
        self.unknown_value = unknown_value
        self.register_buffer("classes", torch.empty(0))

    def _fit(self, input: Tensor) -> None:
        if input.dim() != 1:
            raise ValueError("LabelEncode expects a 1D tensor.")
        if _nan_mask(input).any():
            raise ValueError("LabelEncode cannot fit missing labels.")
        self.classes = torch.unique(input, sorted=True)

    def forward(self, input: Tensor) -> Tensor:
        """Map known labels to class indices."""
        self._check_is_fitted()
        if input.dim() != 1:
            raise ValueError("LabelEncode expects a 1D tensor.")

        if self.classes.numel() == 0:
            raise ValueError("LabelEncode has no fitted classes.")

        matches = input.unsqueeze(1) == self.classes.unsqueeze(0)
        known = matches.any(dim=1) & ~_nan_mask(input)
        if not bool(known.all()):
            raise ValueError("LabelEncode received unknown or missing labels.")

        return matches.to(torch.long).argmax(dim=1)

    def _inverse_transform(self, input: Tensor) -> Tensor:
        if input.dim() != 1:
            raise ValueError("LabelEncode expects a 1D tensor.")

        codes = input.to(torch.long)
        valid = (
            (input >= 0)
            & (codes < self.classes.numel())
            & _integer_code_mask(input, codes)
        )
        if self.classes.dtype == torch.bool and not bool(valid.all()):
            raise ValueError(
                "Cannot inverse-transform unknown values for bool categories."
            )

        output = _unknown_output(
            input.shape,
            template=self.classes,
            unknown_value=self.unknown_value,
        )

        valid = (
            (input >= 0)
            & (codes < self.classes.numel())
            & _integer_code_mask(input, codes)
        )
        output[valid] = self.classes[codes[valid]]
        return output
