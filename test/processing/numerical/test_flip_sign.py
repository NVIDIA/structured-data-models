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


def test_flip_sign_inverse_grouped() -> None:
    table = TableTensor.from_tensor(torch.randn(20, 3))
    processor = FlipSign()
    output = processor.fit_transform_ensemble(
        EnsembleTable.from_table(table, num_members=4)
    )

    # Multiple consecutive outputs per column, e.g., quantiles:
    pred = TableTensor.from_tensor(torch.randn(5, 3 * 9))
    inverse = processor.inverse_transform_ensemble(
        EnsembleTable.from_table(pred, num_members=4)
    )
    for member_id in range(len(output)):
        sign = output[member_id].numerical[0] / table.numerical[0]
        torch.testing.assert_close(
            inverse[member_id].numerical,
            pred.numerical * sign.repeat_interleave(9),
        )
