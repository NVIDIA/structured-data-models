"""Standardization transforms for structured-data feature tensors."""

import torch
from torch import Tensor

from sdm.processing._stats import _constant_feature_mask
from sdm.processing._utils import _as_float
from sdm.processing.base import InvertibleMixin, Processor


class StandardScale(Processor, InvertibleMixin):
    """Center and scale each feature column.

    Constant columns use a unit scale to keep the transform finite and
    invertible.

    Args:
        with_mean: If ``True``, center each column by its fitted mean.
        with_std: If ``True``, scale each column by its fitted standard
            deviation.
    """

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
        input = _as_float(input)
        data_mean = input.mean(dim=0)

        if self.with_mean:
            self.mean = data_mean
        else:
            self.mean = input.new_zeros(input.shape[1])

        if self.with_std and input.size(0) > 1:
            var = input.var(dim=0, correction=0)
            scale = var.sqrt()
            scale[_constant_feature_mask(var, data_mean, input.shape[0])] = 1.0
            self.scale = scale
        else:
            self.scale = input.new_ones(input.shape[1])

    def forward(self, input: Tensor) -> Tensor:
        """Transform ``input`` using the fitted mean and scale.

        Args:
            input: Feature tensor with shape ``[N, C]``.

        Returns:
            Tensor with shape ``[N, C]``.
        """
        return (_as_float(input) - self.mean) / self.scale

    def _inverse_transform(self, input: Tensor) -> Tensor:
        return _as_float(input) * self.scale + self.mean
