# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import FlipSign


def test_flip_sign() -> None:
    table = TableTensor.from_tensor(torch.randn(2, 20, 64))
    ensemble = EnsembleTable.from_table(table, num_members=8)
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(42),
    )
    signs = torch.stack(
        [
            output[member_id].numerical[..., :1, :]
            / table.numerical[..., :1, :]
            for member_id in range(len(output))
        ]
    )
    assert signs.size() == (8, 2, 1, 64)
    for member_id in range(len(output)):
        assert output[member_id].numerical.equal(
            table.numerical * signs[member_id]
        )
    assert not (signs == signs[0]).all()
