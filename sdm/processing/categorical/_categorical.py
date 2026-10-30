# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.nn._buffer import BufferList
from sdm.tensor.categorical import _iter_categories


def _check_categorical_codes(table: TableTensor) -> None:
    """Raise if a categorical code exceeds its column's category vocabulary.

    Negative codes encode missing values and are allowed.

    Args:
        table: The table whose categorical codes are validated.
    """
    if os.getenv("SDM_ENABLE_CATEGORY_CHECKS", "1") == "0":
        return
    columns = table.categorical_columns
    categorical = table.categorical
    for index, category in enumerate(_iter_categories(categorical)):
        codes = categorical.code[..., index]
        message = (
            f"Categorical column {columns[index]!r} contains a code "
            "outside its category vocabulary."
        )
        if torch.compiler.is_compiling():
            valid = (codes < category.numel()).all().to("cpu")
            torch._assert_async(valid, message)
        elif (codes >= category.numel()).any():
            raise ValueError(message)


def _check_categories(
    table: TableTensor,
    categories: BufferList[Tensor],
) -> None:
    """Raise if a category vocabulary differs from the fitted one.

    Args:
        table: The table whose category vocabularies are validated.
        categories: The fitted category vocabulary of every column.
    """
    if os.getenv("SDM_ENABLE_CATEGORY_CHECKS", "1") == "0":
        return
    columns = table.categorical_columns
    categorical = table.categorical
    if categorical.code.size(-1) != len(categories):
        raise ValueError(
            f"Expected {len(categories)} fitted categorical "
            f"columns (got {len(columns)})."
        )
    for index, (actual, expected) in enumerate(
        zip(_iter_categories(categorical), categories)
    ):
        # 'AlignCategories' outputs the fitted vocabulary objects themselves,
        # so skip the comparison, which compiled code cannot evaluate.
        if actual is expected:
            continue
        expected = expected.to(device=actual.device)
        equal = actual.equal(expected)
        if (
            not equal
            and actual.shape == expected.shape
            and actual.is_floating_point()
            and expected.is_floating_point()
        ):
            equal = (
                (actual == expected) | (actual.isnan() & expected.isnan())
            ).all()
        if not equal:
            raise ValueError(
                "Expected the category vocabulary for categorical column "
                f"{columns[index]!r} to match the fitted values and order. "
                "Use 'AlignCategories' before this processor for "
                "independently tensorized inputs."
            )
