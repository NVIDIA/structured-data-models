# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import accumulate

import torch

from sdm import CategoricalTensor, TableTensor, VarLenTensor


def _unique_values(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not values.dtype.is_floating_point:
        return values.unique(return_inverse=True)
    keys = torch.stack((values, values.signbit().to(values.dtype)), dim=-1)
    unique, inverse = keys.unique(dim=0, return_inverse=True)
    return unique[:, 0], inverse


def category_values(
    categorical: CategoricalTensor, dtype: torch.dtype
) -> torch.Tensor:
    """Return numerical category values, or codes for string categories."""
    code = categorical.code
    categories = tuple(
        torch.arange(len(category), device=code.device)
        if isinstance(category, VarLenTensor)
        else category
        for category in categorical.categories
    )
    if not categories:
        return code.to(dtype)

    nan = torch.full((1,), torch.nan, dtype=dtype, device=code.device)
    values = torch.cat(
        [
            value
            for category in categories
            for value in (
                nan,
                category.to(device=code.device, dtype=dtype),
            )
        ]
    )
    sizes = (len(category) + 1 for category in categories)
    start = torch.tensor(
        [*accumulate(sizes, initial=1)][:-1], device=code.device
    )
    index = code.clamp_min(-1) + start
    return values[index].masked_fill(code < 0, torch.nan)


def feature_order(table: TableTensor) -> list[int]:
    """Return block positions in synthetic column order, or block order."""
    # SCMPrior column names encode feature order across semantic blocks.
    names = (*table.numerical_columns, *table.categorical_columns)
    if all(name.startswith("x_") and name[2:].isdigit() for name in names):
        return sorted(range(len(names)), key=lambda i: int(names[i][2:]))
    return list(range(len(names)))


def feature_values(table: TableTensor) -> torch.Tensor:
    """Decode numerical and categorical cells in synthetic column order."""
    return torch.cat(
        (
            table.numerical,
            category_values(table.categorical, table.numerical.dtype),
        ),
        dim=-1,
    )[..., feature_order(table)]


def block_order(table: TableTensor) -> list[int]:
    """Return the positions that restore numerical/categorical block order."""
    order = feature_order(table)
    return sorted(range(len(order)), key=order.__getitem__)


def categorical_mask(table: TableTensor) -> torch.Tensor:
    """Return categorical flags in the same order as decoded cell values."""
    return torch.tensor(
        feature_order(table), device=table.device
    ) >= table.numerical.size(-1)


def mask_cells(table: TableTensor, missing: torch.Tensor) -> TableTensor:
    """Mask numerical and categorical cells in a table."""
    missing = missing[..., block_order(table)]
    num_numerical = table.numerical.size(-1)
    categorical = CategoricalTensor(
        code=table.categorical.code.masked_fill(
            mask=missing[..., num_numerical:], value=-1
        ),
        categories=table.categorical.categories,
    )
    return table.replace_blocks(
        numerical=table.numerical.masked_fill(
            mask=missing[..., :num_numerical], value=torch.nan
        ),
        categorical=categorical,
    )
