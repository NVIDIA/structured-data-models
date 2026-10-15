# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import (
    MaskBlock,
    MaskByTarget,
    MaskCategorical,
    MaskCensoring,
    MaskRows,
)


def _numerical_table() -> TableTensor:
    return TableTensor(
        columns={Stype.numerical: [*(f"x{i}" for i in range(6)), "target"]},
        numerical=torch.randn(200, 7),
    )


def test_mask_rows_masks_the_same_rows_in_a_fraction_of_columns() -> None:
    table = _numerical_table()

    output = MaskRows(
        rows=(0.25, 0.25), columns=(0.5, 0.5), target="target"
    ).fit_transform(table, generator=torch.Generator().manual_seed(0))

    missing = output.numerical.isnan()
    selected = missing.any(dim=-2)
    assert selected.sum() == 3
    assert missing[:, selected].eq(missing[:, selected][:, :1]).all()
    assert missing[:, selected][:, 0].sum() == 50
    assert output.numerical[:, -1].equal(table.numerical[:, -1])


def test_mask_block_masks_the_same_rows_in_a_fixed_number_of_columns() -> None:
    table = _numerical_table()

    output = MaskBlock(
        rows=(0.2, 0.2), num_columns=(2, 2), target="target"
    ).fit_transform(table, generator=torch.Generator().manual_seed(1))

    missing = output.numerical.isnan()
    selected = missing.any(dim=-2)
    assert selected.sum() == 2
    assert missing[:, selected].eq(missing[:, selected][:, :1]).all()
    assert missing[:, selected][:, 0].sum() == 40
    assert output.numerical[:, -1].equal(table.numerical[:, -1])


def test_mask_by_target_masks_rows_of_one_categorical_class() -> None:
    target = torch.arange(200).remainder(4)
    table = TableTensor(
        columns={
            Stype.numerical: [f"x{i}" for i in range(4)],
            Stype.categorical: ["target"],
        },
        numerical=torch.randn(200, 4),
        categorical=CategoricalTensor(
            target[:, None], categories=(torch.arange(4),)
        ),
    )

    output = MaskByTarget("target", num_columns=(2, 2)).fit_transform(
        table, generator=torch.Generator().manual_seed(2)
    )

    missing = output.numerical.isnan()
    masked_rows = missing.any(dim=-1)
    assert missing.any(dim=-2).sum() == 2
    assert masked_rows.sum() == 50
    assert target[masked_rows].unique().numel() == 1
    assert missing[:, missing.any(dim=-2)].eq(masked_rows[:, None]).all()
    assert output.categorical.equal(table.categorical)


def test_mask_categorical_masks_requested_categorical_columns() -> None:
    code = torch.randint(3, (200, 4))
    table = TableTensor(
        columns={
            Stype.numerical: ["x0", "x1"],
            Stype.categorical: ["c0", "c1", "c2", "target"],
        },
        numerical=torch.randn(200, 2),
        categorical=CategoricalTensor(code, categories=(torch.arange(3),) * 4),
    )

    output = MaskCategorical(
        rows=(0.25, 0.25), num_columns=(2, 2), target="target"
    ).fit_transform(table, generator=torch.Generator().manual_seed(3))

    missing = output.categorical.code < 0
    selected = missing.any(dim=-2)
    assert output.numerical.equal(table.numerical)
    assert selected.sum() == 2
    assert missing[:, selected].sum(dim=-2).eq(50).all()
    assert output.categorical.code[:, -1].equal(table.categorical.code[:, -1])


def test_mask_censoring_masks_one_end_of_a_column() -> None:
    values = torch.arange(200, dtype=torch.float32)
    table = TableTensor.from_tensor(values[:, None])

    output = MaskCensoring(
        rows=(0.25, 0.25), num_columns=(1, 1)
    ).fit_transform(table, generator=torch.Generator().manual_seed(4))

    missing = output.numerical[:, 0].isnan()
    assert missing.sum() == 50
    masked = values[missing]
    observed = values[~missing]
    assert masked.max() < observed.min() or masked.min() > observed.max()
