from __future__ import annotations

import torch
from torch import Tensor

from schemafm.processing._stats import _constant_feature_mask
from schemafm.processing.base import InvertibleMixin, Processor


class StandardScale(Processor, InvertibleMixin):
    """Center and scale each feature column."""

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _fit(self, input: Tensor) -> None:
        if self.with_mean:
            self.mean = input.mean(dim=0)
        else:
            self.mean = input.new_zeros(input.shape[1])

        if self.with_std and input.size(0) > 1:
            scale = input.std(dim=0, correction=0)
            var = input.var(dim=0, correction=0)
            mean = input.mean(dim=0)
            scale[_constant_feature_mask(var, mean, input.shape[0])] = 1.0
            self.scale = scale
        else:
            self.scale = input.new_ones(input.shape[1])

    def forward(self, input: Tensor) -> Tensor:
        """Transform ``input`` using the fitted mean and scale."""
        self._check_is_fitted()
        return (input - self.mean) / self.scale

    def _inverse_transform(self, input: Tensor) -> Tensor:
        return input * self.scale + self.mean
