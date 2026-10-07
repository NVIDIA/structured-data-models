# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import ColumnarTensor, TableTensor


def test_tensor_cache_metadata() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    base = TableTensor.from_tensor(values)
    changed_data = TableTensor.from_tensor(values + 1)
    base_hash = base._stable_hash_for_caching()
    assert base_hash == changed_data._stable_hash_for_caching()

    changed_layout = TableTensor.from_tensor(values.T.contiguous().T)
    changed_dtype = TableTensor.from_tensor(values.double())
    changed_schema = TableTensor(
        numerical=values, columns={"numerical": ("w", "x", "y", "z")}
    )
    for changed in (changed_layout, changed_dtype, changed_schema):
        assert base_hash != changed._stable_hash_for_caching()


def test_tensor_cache_leaf_aliases() -> None:
    values = torch.arange(3)
    shared = ColumnarTensor((values, values))
    separate = ColumnarTensor((values, values.clone()))
    assert (
        shared._stable_hash_for_caching()
        != separate._stable_hash_for_caching()
    )
