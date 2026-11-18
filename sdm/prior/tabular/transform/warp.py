# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._scale import _standardize


def _log_uniform(
    size: tuple[int, ...],
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    log_low = math.log(low)
    log_high = math.log(high)
    u = torch.rand(size, device=device, generator=generator, dtype=dtype)
    return torch.exp(u * (log_high - log_low) + log_low)


class WarpTargetTails(PriorTransform):
    """Apply a random power or exponential warp to a numerical target.

    Standardize the target, apply an asymmetric signed power or signed
    exponential map, and clip the result to ``[-100, 100]``.
    """

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        if y.numerical.size(-1) == 0:
            return x, y
        values = _standardize(y.numerical)
        num_columns = values.size(-1)
        device, dtype = values.device, values.dtype

        torch.rand(num_columns, device=device, generator=generator)

        p_pos = _log_uniform(
            (num_columns,),
            low=0.05,
            high=5.0,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        p_neg = _log_uniform(
            (num_columns,),
            low=0.05,
            high=5.0,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        p_pos = p_pos.view(*([1] * (values.dim() - 1)), num_columns)
        p_neg = p_neg.view(*([1] * (values.dim() - 1)), num_columns)
        power_warp = torch.where(
            condition=values >= 0,
            input=values.abs().pow(p_pos),
            other=-values.abs().pow(p_neg),
        )

        scale = _log_uniform(
            (num_columns,),
            low=0.3,
            high=2.0,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        scale = scale.view(*([1] * (values.dim() - 1)), num_columns)
        exp_warp = values.sign() * torch.expm1(values.abs() * scale)

        kind = torch.randint(
            low=0,
            high=2,
            size=(num_columns,),
            device=device,
            generator=generator,
        ).bool()
        kind = kind.view(*([1] * (values.dim() - 1)), num_columns)
        warped = torch.where(kind, exp_warp, power_warp).clamp(-100.0, 100.0)
        return x, y.replace_blocks(numerical=warped)
