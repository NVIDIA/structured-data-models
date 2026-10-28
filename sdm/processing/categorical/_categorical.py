# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import cast

import torch
from torch import Tensor

from sdm import NullableTensor, StringTensor, TableTensor, VarLenTensor
from sdm.nn._buffer import BufferList
from sdm.tensor.categorical import _iter_categories
from sdm.tensor.var_len import _contiguous

_CATEGORY_TYPES = (Tensor, NullableTensor, VarLenTensor, StringTensor)


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
            torch._assert_async(
                _check_category_code_bounds(codes, category.numel(), message),
                message,
            )
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
        expected = expected.to(device=actual.device)
        message = (
            "Expected the category vocabulary for categorical column "
            f"{columns[index]!r} to match the fitted values and order. "
            "Use 'AlignCategories' before this processor for "
            "independently tensorized inputs."
        )
        if torch.compiler.is_compiling():
            actual_tensors, actual_layout = _pack_category(actual)
            expected_tensors, expected_layout = _pack_category(expected)
            # The message overload keeps this runtime check in the graph.
            torch._assert_async(
                _check_category_values(
                    actual=actual_tensors,
                    actual_layout=actual_layout,
                    expected=expected_tensors,
                    expected_layout=expected_layout,
                    message=message,
                ),
                message,
            )
        elif not actual.equal(expected):
            raise ValueError(message)


def _pack_category(category: Tensor) -> tuple[list[Tensor | None], list[int]]:
    """Pack vocabulary data for the operator without tensor wrappers."""
    kind = _CATEGORY_TYPES.index(type(category))
    if isinstance(category, VarLenTensor):
        # Keep the base offsets and logical view metadata across op inputs.
        return [category._data, category._offset, category._valid], [
            kind,
            category.numel(),
            category.stride(0),
            category._logical_storage_offset,
        ]
    if isinstance(category, NullableTensor):
        return [category._data, category._valid], [kind]
    return [category], [kind]


def _unpack_category(
    tensors: list[Tensor | None], layout: list[int]
) -> Tensor:
    category_type = _CATEGORY_TYPES[layout[0]]
    if category_type is Tensor:
        return cast(Tensor, tensors[0])
    if category_type is NullableTensor:
        return NullableTensor(
            cast(Tensor, tensors[0]), cast(Tensor, tensors[1])
        )
    category = cast(
        VarLenTensor,
        category_type(
            data=cast(Tensor, tensors[0]),
            offset=cast(Tensor, tensors[1]),
            valid=tensors[2],
            size=(layout[1],),
            stride=(layout[2],),
            storage_offset=layout[3],
        ),
    )
    # Custom-op kernels run below Python dispatch: compact views directly.
    return category if category.is_contiguous() else _contiguous(category)


@torch.library.custom_op("sdm::check_category_values", mutates_args=())
def _check_category_values(
    actual: list[Tensor | None],
    actual_layout: list[int],
    expected: list[Tensor | None],
    expected_layout: list[int],
    message: str,
) -> Tensor:
    actual_category = _unpack_category(actual, actual_layout)
    expected_category = _unpack_category(expected, expected_layout)
    if not actual_category.equal(expected_category):
        raise ValueError(message)
    return torch.ones((), dtype=torch.bool, device="cpu")


@_check_category_values.register_fake
def _check_category_values_fake(
    actual: list[Tensor | None],
    actual_layout: list[int],
    expected: list[Tensor | None],
    expected_layout: list[int],
    message: str,
) -> Tensor:
    return torch.empty((), dtype=torch.bool, device="cpu")


@torch.library.custom_op("sdm::check_category_code_bounds", mutates_args=())
def _check_category_code_bounds(
    codes: Tensor, num_categories: int, message: str
) -> Tensor:
    if (codes >= num_categories).any():
        raise ValueError(message)
    return torch.ones((), dtype=torch.bool, device="cpu")


@_check_category_code_bounds.register_fake
def _check_category_code_bounds_fake(
    codes: Tensor, num_categories: int, message: str
) -> Tensor:
    return torch.empty((), dtype=torch.bool, device="cpu")
