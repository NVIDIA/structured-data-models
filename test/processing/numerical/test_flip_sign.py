# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
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


@pytest.mark.parametrize(
    ("locations", "expected_num_groups"),
    [
        pytest.param(
            tuple((0, i) for i in range(4)), 1, id="canonical_positions"
        ),
        pytest.param(((0, 1), (0, 0)), 1, id="reordered_positions"),
    ],
)
def test_flip_sign_group_preservation(
    locations: tuple[tuple[int, int], ...],
    expected_num_groups: int,
) -> None:
    # Any position permutation stays one group; only sharing (test_flip_sign)
    # falls back to a group per member.
    num_rows = max(position for _, position in locations) + 1
    table = TableTensor.from_tensor(torch.randn(num_rows, 4, 3))
    ensemble = EnsembleTable(groups=(table,), locations=locations)
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.num_groups == expected_num_groups
    for member_id, (_, position) in enumerate(locations):
        expected = table.numerical[position] * processor._signs[member_id]
        torch.testing.assert_close(output[member_id].numerical, expected)

    restored = processor.inverse_transform_ensemble(output)
    for member_id, (_, position) in enumerate(locations):
        torch.testing.assert_close(
            restored[member_id].numerical, table.numerical[position]
        )
