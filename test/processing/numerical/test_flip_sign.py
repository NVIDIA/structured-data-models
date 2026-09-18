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


def test_flip_sign_preserves_group_when_members_already_own_a_position() -> (
    None
):
    # One group stacking 4 already-distinct member rows, as produced when
    # ensemble members diverge upstream (e.g. after per-member subsampling)
    # rather than sharing a single stored row.
    table = TableTensor.from_tensor(torch.randn(4, 5, 3))
    ensemble = EnsembleTable(
        groups=(table,), locations=tuple((0, i) for i in range(4))
    )
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(0),
    )

    # The forward transform must not split the shared physical group into
    # one group per member: doing so breaks position-dependent fitted
    # processors (e.g. `Standardize`) placed around `FlipSign` in a
    # `Sequential`, since they rely on a stable group/position layout
    # between `transform` and `inverse_transform`.
    assert output.num_groups == 1
    assert output._locations == ensemble._locations

    signs = torch.stack(
        [
            output[member_id].numerical / table.numerical[member_id]
            for member_id in range(len(output))
        ]
    )
    assert signs.abs().equal(torch.ones_like(signs))
    assert not (signs == signs[0]).all()

    restored = processor.inverse_transform_ensemble(output)
    assert restored.num_groups == 1
    assert restored._locations == ensemble._locations
    for member_id in range(len(restored)):
        assert restored[member_id].numerical.equal(table.numerical[member_id])
