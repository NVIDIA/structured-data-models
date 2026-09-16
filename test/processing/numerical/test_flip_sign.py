# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import Choice, FlipSign, Identity


def test_flip_sign_draws_independent_member_signs() -> None:
    table = TableTensor.from_tensor(torch.randn(2, 20, 64))
    ensemble = EnsembleTable.from_table(table, num_members=8)
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(42),
    )
    signs = torch.stack(
        [
            output.table(member_id).numerical[..., :1, :]
            / table.numerical[..., :1, :]
            for member_id in range(output.num_members)
        ]
    )
    assert signs.size() == (8, 2, 1, 64)
    for member_id in range(output.num_members):
        assert output.table(member_id).numerical.equal(
            table.numerical * signs[member_id]
        )
    assert not (signs == signs[0]).all()

    inverted = processor.inverse_transform_ensemble(output)
    for member_id in range(inverted.num_members):
        assert inverted.table(member_id).equal(table)

    repeated = FlipSign().fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(42),
    )
    for member_id in range(repeated.num_members):
        assert repeated.table(member_id).equal(output.table(member_id))

    query = TableTensor.from_tensor(torch.full((2, 5, 64), 2.0))
    query_output = processor.transform_ensemble(
        EnsembleTable.from_table(query, num_members=8)
    )
    for member_id in range(query_output.num_members):
        assert query_output.table(member_id).numerical.equal(
            query.numerical * signs[member_id]
        )


def test_flip_sign_inside_choice() -> None:
    table = TableTensor.from_tensor(torch.randn(20, 64))
    ensemble = EnsembleTable.from_table(table, num_members=4)
    processor = Choice(FlipSign(), Identity(), method="round_robin")

    output = processor.fit_transform_ensemble(ensemble)
    for member_id in (1, 3):
        assert output.table(member_id).equal(table)
    for member_id in (0, 2):
        numerical = output.table(member_id).numerical
        signs = numerical[..., :1, :] / table.numerical[..., :1, :]
        assert numerical.equal(table.numerical * signs)
