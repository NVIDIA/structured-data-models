# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import WarpTargetTails


def test_warp_target_tails_standardizes_and_warps_reproducibly() -> None:
    x = TableTensor.from_tensor(torch.zeros(257, 3))
    values = torch.linspace(-3.0, 3.0, 257, dtype=torch.float64)[:, None]
    y = TableTensor.from_tensor(values)
    shifted = TableTensor.from_tensor(7.0 * values + 42.0)
    transform = WarpTargetTails()

    output_x, output_y = transform(
        x=x, y=y, generator=torch.Generator().manual_seed(0)
    )
    _, shifted_y = transform(
        x=x, y=shifted, generator=torch.Generator().manual_seed(0)
    )

    assert output_x.numerical.equal(x.numerical)
    standardized = (values - values.mean()) / values.std(correction=1)
    assert not output_y.numerical.equal(standardized)
    assert output_y.numerical.dtype == values.dtype
    assert output_y.numerical.isfinite().all()
    assert output_y.numerical.abs().max() <= 100.0
    assert output_y.numerical[128].eq(0).all()
    assert output_y.numerical.diff(dim=-2).ge(0).all()
    assert torch.allclose(
        output_y.numerical, shifted_y.numerical, atol=1e-5, rtol=1e-5
    )


def test_warp_target_tails_keeps_a_categorical_target() -> None:
    x = TableTensor.from_tensor(torch.zeros(8, 3))
    code = torch.randint(3, (8, 1))
    y = TableTensor(
        categorical=CategoricalTensor(
            code=code, categories=(torch.arange(3),)
        ),
    )

    _, output_y = WarpTargetTails()(x, y)

    assert output_y.categorical.code.equal(code)
