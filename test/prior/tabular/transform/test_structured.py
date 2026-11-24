# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import (
    MaskBlock,
    MaskByTarget,
    MaskCategorical,
    MaskCensoring,
    MaskRows,
    MaskStructured,
    PriorTransform,
)


def _numerical_task() -> tuple[TableTensor, TableTensor]:
    x = TableTensor(numerical=torch.randn(200, 6))
    y = TableTensor.from_tensor(torch.randn(200, 1))
    return x, y


def test_mask_rows_masks_the_same_rows_in_a_fraction_of_columns() -> None:
    x, y = _numerical_task()

    output, output_y = MaskRows(rows=(0.25, 0.25), columns=(0.5, 0.5))(x, y)

    missing = output.numerical.isnan()
    selected = missing.any(dim=-2)
    assert selected.sum() == 3
    assert missing[:, selected].eq(missing[:, selected][:, :1]).all()
    assert missing[:, selected][:, 0].sum() == 50
    assert output_y.numerical.equal(y.numerical)


def test_mask_block_masks_the_same_rows_in_a_fixed_number_of_columns() -> None:
    x, y = _numerical_task()

    output, _ = MaskBlock(rows=(0.2, 0.2), num_columns=(2, 2))(x, y)

    missing = output.numerical.isnan()
    selected = missing.any(dim=-2)
    assert selected.sum() == 2
    assert missing[:, selected].eq(missing[:, selected][:, :1]).all()
    assert missing[:, selected][:, 0].sum() == 40


def test_mask_by_target_masks_rows_of_one_categorical_class() -> None:
    x = TableTensor.from_tensor(torch.randn(200, 4))
    target = torch.arange(200).remainder(4)
    y = TableTensor(
        categorical=CategoricalTensor(
            code=target[:, None], categories=(torch.arange(4),)
        ),
    )

    output, output_y = MaskByTarget(num_columns=(2, 2))(x, y)

    missing = output.numerical.isnan()
    masked_rows = missing.any(dim=-1)
    assert missing.any(dim=-2).sum() == 2
    assert masked_rows.sum() == 50
    assert target[masked_rows].unique().numel() == 1
    assert missing[:, missing.any(dim=-2)].eq(masked_rows[:, None]).all()
    assert output_y.categorical.code.equal(y.categorical.code)


def test_mask_categorical_masks_requested_categorical_columns() -> None:
    x = TableTensor(
        numerical=torch.randn(200, 2),
        categorical=CategoricalTensor(
            code=torch.randint(3, (200, 3)),
            categories=(torch.arange(3),) * 3,
        ),
    )
    y = TableTensor.from_tensor(torch.randn(200, 1))

    output, _ = MaskCategorical(rows=(0.25, 0.25), num_columns=(2, 2))(x, y)

    missing = output.categorical.code < 0
    selected = missing.any(dim=-2)
    assert output.numerical.equal(x.numerical)
    assert selected.sum() == 2
    assert missing[:, selected].sum(dim=-2).eq(50).all()


def test_mask_censoring_masks_the_requested_tail_count() -> None:
    values = torch.arange(2, dtype=torch.float32).repeat_interleave(100)
    x = TableTensor.from_tensor(values[:, None])
    y = TableTensor.from_tensor(torch.randn(200, 1))

    output, _ = MaskCensoring(rows=(0.25, 0.25), num_columns=(1, 1))(x, y)

    missing = output.numerical[:, 0].isnan()
    assert missing.sum() == 50
    masked = values[missing]
    assert masked.eq(values.min()).all() or masked.eq(values.max()).all()


@pytest.mark.parametrize(
    "transform",
    [
        MaskByTarget(num_columns=(3, 4)),
        MaskCategorical(num_columns=(3, 4)),
        MaskCensoring(num_columns=(3, 4)),
    ],
)
def test_masks_keep_tables_with_too_few_columns(
    transform: PriorTransform,
) -> None:
    x = TableTensor(
        numerical=torch.randn(8, 1),
        categorical=CategoricalTensor(
            code=torch.randint(3, (8, 1)),
            categories=(torch.arange(3),),
        ),
    )
    y = TableTensor.from_tensor(torch.randn(8, 1))

    result = transform(x, y)
    assert result is not None
    output, _ = result

    assert output.numerical.equal(x.numerical)
    assert output.categorical.code.equal(x.categorical.code)


def test_mask_structured_is_reproducible_and_masks_features() -> None:
    x, y = _numerical_task()
    transform = MaskStructured()

    first, first_y = transform(
        x=x, y=y, generator=torch.Generator().manual_seed(0)
    )
    second, _ = transform(x, y, generator=torch.Generator().manual_seed(0))

    first_missing = first.numerical.isnan()
    assert first_missing.any()
    assert first_missing.equal(second.numerical.isnan())
    assert first_y.numerical.equal(y.numerical)
