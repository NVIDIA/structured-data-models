# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import FlipSign, Sequential, Standardize


def test_flip_sign() -> None:
    table = TableTensor.from_tensor(torch.randn(2, 20, 64))
    ensemble = EnsembleTable.from_table(table, num_members=8)
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(42),
    )
    assert output.num_groups == ensemble.num_groups
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

    restored = processor.inverse_transform_ensemble(output)
    for member_id in range(len(restored)):
        torch.testing.assert_close(
            restored[member_id].numerical,
            table.numerical,
        )


@pytest.mark.parametrize(
    ("group_sizes", "locations"),
    [
        pytest.param(
            (4,),
            tuple((0, i) for i in range(4)),
            id="canonical_positions",
        ),
        pytest.param(
            (2,),
            ((0, 1), (0, 0)),
            id="reordered_positions",
        ),
        pytest.param(
            (3, 2),
            ((0, 2), (1, 0), (0, 0), (1, 1)),
            id="multiple_groups",
        ),
    ],
)
def test_flip_sign_group_preservation(
    group_sizes: tuple[int, ...],
    locations: tuple[tuple[int, int], ...],
) -> None:
    groups = tuple(
        TableTensor.from_tensor(
            torch.arange(size * 4 * 3, dtype=torch.float32).reshape(size, 4, 3)
            + group_id * 100
            + 1
        )
        for group_id, size in enumerate(group_sizes)
    )
    ensemble = EnsembleTable(groups=groups, locations=locations)
    processor = Sequential(Standardize(), FlipSign())

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.num_groups == ensemble.num_groups

    restored = processor.inverse_transform_ensemble(output)
    for member_id in range(len(ensemble)):
        torch.testing.assert_close(
            restored[member_id].numerical,
            ensemble[member_id].numerical,
        )
