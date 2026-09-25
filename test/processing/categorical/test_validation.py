# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, TableTensor
from sdm.testing import withCUDA


def _table(code: torch.Tensor) -> TableTensor:
    return TableTensor(
        categorical=CategoricalTensor(
            code=code,
            categories=tuple(
                torch.arange(size, device=code.device) for size in (2, 4)
            ),
        ),
    )


@withCUDA
@pytest.mark.parametrize(
    "processor_cls", [sp.AddCategoryCounts, sp.ImputeMode]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(4, 2), (2, 3, 4, 2)])
@pytest.mark.parametrize("invalid_columns", [(0,), (1,), (0, 1)])
def test_categorical_processors_reject_out_of_range_codes(
    device: torch.device,
    processor_cls: type[sp.AddCategoryCounts] | type[sp.ImputeMode],
    dtype: torch.dtype,
    shape: tuple[int, ...],
    invalid_columns: tuple[int, ...],
) -> None:
    codes = torch.zeros((*shape[:-1], 4), dtype=dtype, device=device)[..., ::2]
    processor = processor_cls().fit(_table(codes.clone()))
    for column in invalid_columns:
        # Put the first column's error later in the data than the second's.
        position = tuple(size - 1 if column == 0 else 0 for size in shape[:-1])
        codes[(*position, column)] = (2, 4)[column]
    table = _table(codes)
    original = codes.clone()
    message = rf"cat_{invalid_columns[0]}.*outside.*vocabulary"

    with pytest.raises(ValueError, match=message):
        processor_cls().fit(table)
    with pytest.raises(ValueError, match=message):
        processor.transform(table)

    torch.testing.assert_close(codes, original)


@withCUDA
@pytest.mark.parametrize(
    "processor_cls", [sp.AddCategoryCounts, sp.ImputeMode]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_categorical_processors_accept_valid_and_missing_codes(
    device: torch.device,
    processor_cls: type[sp.AddCategoryCounts] | type[sp.ImputeMode],
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
) -> None:
    codes = torch.tensor(
        [[torch.iinfo(dtype).min, -7], [1, 3]],
        dtype=dtype,
        device=device,
    ).expand(*batch_shape, 2, 2)
    original = codes.clone()
    processor = processor_cls().fit(_table(torch.zeros_like(codes)))

    output = processor.transform(_table(codes))

    expected = (
        codes.clamp_min(0) if isinstance(processor, sp.ImputeMode) else codes
    )
    torch.testing.assert_close(output.categorical.code, expected)
    torch.testing.assert_close(codes, original)


@withCUDA
@pytest.mark.parametrize(
    "processor_cls", [sp.AddCategoryCounts, sp.ImputeMode]
)
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_categorical_processors_accept_empty_queries(
    device: torch.device,
    processor_cls: type[sp.AddCategoryCounts] | type[sp.ImputeMode],
    batch_shape: tuple[int, ...],
) -> None:
    context = _table(
        torch.zeros((*batch_shape, 4, 2), dtype=torch.int32, device=device)
    )
    query = _table(
        torch.empty((*batch_shape, 0, 2), dtype=torch.int32, device=device)
    )

    output = processor_cls().fit(context).transform(query)

    assert output.categorical.equal(query.categorical)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_category_counts_accepts_empty_vocabularies_and_rejects_codes(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    codes = torch.full((3, 1), -7, dtype=dtype, device=device)
    table = TableTensor(
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.empty(0, device=device),),
        ),
    )

    output = sp.AddCategoryCounts().fit_transform(table)

    assert output.categorical.equal(table.categorical)
    codes[-1, 0] = 0
    with pytest.raises(ValueError, match=r"cat_0.*outside.*vocabulary"):
        sp.AddCategoryCounts().fit(table)


@withCUDA
@pytest.mark.parametrize("shape", [(0, 2), (2, 0, 2), (0, 3, 2)])
def test_category_counts_accepts_empty_contexts(
    device: torch.device,
    shape: tuple[int, ...],
) -> None:
    table = _table(torch.empty(shape, dtype=torch.int32, device=device))

    output = sp.AddCategoryCounts().fit_transform(table)

    assert output.categorical.equal(table.categorical)
