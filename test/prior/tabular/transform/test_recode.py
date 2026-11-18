# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.prior.tabular.transform import RecodeHighCardinality


def test_recode_high_cardinality_preserves_nonfinite_columns() -> None:
    numerical = torch.randn(300, 3)
    numerical[0, 0] = torch.nan
    categorical = CategoricalTensor(
        code=torch.randint(50, (300, 2)),
        categories=(torch.arange(50),) * 2,
    )
    x = TableTensor(
        columns={
            Stype.numerical: ["n0", "n1", "n2"],
            Stype.categorical: ["c0", "c1"],
        },
        numerical=numerical,
        categorical=categorical,
    )
    y = TableTensor.from_tensor(torch.randn(300, 1))

    output, output_y = RecodeHighCardinality(
        min_levels=20,
        num_columns=(5, 5),
        categorical=True,
        standardize=True,
    )(x=x, y=y)

    assert output.columns[Stype.numerical] == ("n0",)
    torch.testing.assert_close(
        output.numerical, numerical[:, :1], equal_nan=True
    )
    assert set(output.columns[Stype.categorical]) == {"n1", "n2", "c0", "c1"}
    for index, category in enumerate(output.categorical.categories):
        code = output.categorical.code[:, index]
        assert 20 <= code[code >= 0].unique().numel() <= 300
        values = category[code[code >= 0]]
        torch.testing.assert_close(values.mean(), values.new_tensor(0.0))
        torch.testing.assert_close(values.std(), values.new_tensor(1.0))
    assert output_y.numerical.equal(y.numerical)


def test_recode_keeps_datasets_with_too_few_rows() -> None:
    x = TableTensor.from_tensor(torch.randn(200, 3))
    y = TableTensor.from_tensor(torch.randn(200, 1))

    output, target = RecodeHighCardinality()(x, y)

    assert output.numerical.equal(x.numerical)
    assert target.numerical.equal(y.numerical)
