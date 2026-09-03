# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import ColumnarTensor, TableTensor
from sdm.relational.join import join_index
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_join_index(dtype: torch.dtype, device: torch.device) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(8, device=device),)),
    )

    left_index, right_index = join_index(
        left_table=left_table,
        right_table=right_table,
        left_keys=["id"],
        right_keys=["id"],
        dtype=dtype,
    )

    assert left_index.dtype == dtype
    assert right_index.dtype == dtype
    assert left_index.device == device
    assert right_index.device == device
    assert left_index.sort()[0].equal(torch.arange(8, device=device))
    assert right_index.equal(left_index)


def test_join_index_cast() -> None:
    left_table = TableTensor(
        columns={"id": ("friend",)},
        id=ColumnarTensor((torch.tensor([1.0, float("nan"), 3.0]),)),
    )
    right_table = TableTensor(
        columns={"id": ("user_id",)},
        id=ColumnarTensor((torch.tensor([1, 2, 3]),)),
    )

    left_index, right_index = join_index(
        left_table=left_table,
        right_table=right_table,
        left_keys=["friend"],
        right_keys=["user_id"],
    )

    assert left_index.equal(torch.tensor([0, 2]))
    assert right_index.equal(torch.tensor([0, 2]))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_invalid_dtype(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(8, device=device),)),
    )

    with pytest.raises(TypeError, match="requires an integer input type"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=dtype,
        )


@withCUDA
def test_overflow(device: torch.device) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(200, device=device),)),
    )

    with pytest.raises(ValueError, match="row indices up to 199"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=torch.int8,
        )
