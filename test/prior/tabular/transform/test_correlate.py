# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import CorrelateColumns


def test_correlate_columns_couples_numerical_columns() -> None:
    x = TableTensor(numerical=torch.randn(200, 5))
    y = TableTensor.from_tensor(torch.randn(200, 1))

    output, output_y = CorrelateColumns(num_blocks=1, rho=0.9)(x, y)

    changed = output.numerical.ne(x.numerical).any(dim=0)
    assert changed.sum() >= 2
    correlation = output.numerical[:, changed].mT.corrcoef()
    off_diagonal = ~torch.eye(correlation.size(0), dtype=torch.bool)
    assert correlation[off_diagonal].gt(0.9).all()
    assert output_y.numerical.equal(y.numerical)


def test_correlate_columns_keeps_categories_and_missing_cells() -> None:
    code = torch.randint(4, (200, 2))
    code[:20] = -1
    x = TableTensor(
        categorical=CategoricalTensor(
            code=code, categories=(torch.arange(4),) * 2
        ),
    )
    y = TableTensor.from_tensor(torch.randn(200, 1))

    output, _ = CorrelateColumns(num_blocks=1, rho=0.9)(x, y)

    assert output.categorical.code.sort(dim=0).values.equal(
        code.sort(dim=0).values
    )
    assert output.categorical.code.eq(-1).equal(code.eq(-1))
