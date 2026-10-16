# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor


class AnomalyPreprocess(Processor):
    """Match Kumo-Anomaly's feature width and min-max preprocessing.

    Numerical inputs wider than the checkpoint are centered and projected
    with full-SVD PCA, using the largest-loading-positive sign convention.
    Narrower inputs are right-padded with zeros after scaling. Constant
    columns use a range of one. Query values are not clipped to the fitted
    range. Non-numerical columns are not model inputs.

    Fit on reference data for held-out evaluation. Fitting on the entire
    scored series is the original SDK's explicitly transductive workflow.
    Inputs must be finite. Rows and leading batch dimensions are preserved;
    statistics are fitted independently over the row dimension.

    Args:
        num_features: Feature width required by the checkpoint.
        scale_factor: Multiplier applied after min-max normalization.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    mean: Tensor
    components: Tensor
    minimum: Tensor
    scale: Tensor

    def __init__(
        self, num_features: int = 40, scale_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.scale_factor = scale_factor
        for name in ("mean", "components", "minimum", "scale"):
            self.register_buffer(name, torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Learn the projection and ranges using reference rows only."""
        x = table.numerical
        self._set_buffer("mean", x.new_zeros((*x.shape[:-2], 1, x.size(-1))))
        self._set_buffer("components", x.new_empty(0))
        if x.size(-1) > self.num_features:
            if x.size(-2) < self.num_features:
                raise ValueError(
                    "PCA requires at least 'num_features' reference rows"
                )
            self._set_buffer("mean", x.mean(dim=-2, keepdim=True))
            _, _, vh = torch.linalg.svd(x - self.mean, full_matrices=False)
            vh = vh[..., : self.num_features, :]
            signs = vh.gather(-1, vh.abs().argmax(dim=-1, keepdim=True)).sign()
            self._set_buffer("components", (vh * signs).transpose(-1, -2))
            x = (x - self.mean) @ self.components
        minimum = x.amin(dim=-2, keepdim=True)
        scale = x.amax(dim=-2, keepdim=True) - minimum
        self._set_buffer("minimum", minimum)
        self._set_buffer("scale", torch.where(scale == 0, 1, scale))

    def _transform(self, table: TableTensor) -> TableTensor:
        """Apply fitted preprocessing and append unscored padding columns."""
        x = table.numerical
        if x.size(-1) != self.mean.size(-1):
            raise ValueError("Reference and query feature counts must match")
        if self.components.numel():
            x = (x - self.mean) @ self.components
        x = (x - self.minimum) / self.scale * self.scale_factor
        x = F.pad(x, (0, self.num_features - x.size(-1)))
        # Model-space columns are not raw features when PCA is used.
        return TableTensor(
            columns={
                Stype.numerical: [
                    f"feature_{i}" for i in range(self.num_features)
                ]
            },
            numerical=x,
        )
