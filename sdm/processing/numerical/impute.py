# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _isfinite


class ImputeMean(Processor):
    """Replace missing numerical values with fitted per-column means.

    Infinite values are ignored when fitting the mean but preserved during
    transformation.

    Args:
        fill_value: Finite value used for columns whose fitted mean is
            undefined (columns without finite values).
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _mean: Tensor

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.register_buffer("_mean", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        mean = torch.nanmean(
            numerical.masked_fill(~_isfinite(numerical), torch.nan),
            dim=-2,
            keepdim=True,
        )
        self._set_buffer(
            name="_mean",
            value=torch.where(mean.isnan(), self.fill_value, mean),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Replace NaNs with the fitted per-column means."""
        numerical = table.numerical
        numerical = torch.where(numerical.isnan(), self._mean, numerical)
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        if self.fill_value == 0.0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"fill_value={self.fill_value!r})"
        )


class ImputeConstant(Processor, InvertibleMixin):
    """Replace missing numerical values with a constant.

    Infinite values are preserved. Inverse transformation leaves values
    unchanged.

    Args:
        fill_value: Value used to replace NaNs.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def __init__(self, fill_value: float) -> None:
        super().__init__()
        self.fill_value = fill_value

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(
            numerical=table.numerical.nan_to_num(
                nan=self.fill_value,
                posinf=torch.inf,
                neginf=-torch.inf,
            ),
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"fill_value={self.fill_value!r})"
        )
