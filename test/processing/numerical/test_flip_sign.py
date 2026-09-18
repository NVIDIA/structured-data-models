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


def test_flip_sign_falls_back_when_positions_are_reordered() -> None:
    # Same group size as member count, but member 0 sits at position 1 and
    # member 1 sits at position 0
    table = TableTensor.from_tensor(torch.randn(2, 4, 3))
    ensemble = EnsembleTable(groups=(table,), locations=((0, 1), (0, 0)))
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(7),
    )

    assert output.num_groups == 2
    for member_id, position in ((0, 1), (1, 0)):
        expected = table.numerical[position] * processor._signs[member_id]
        assert output[member_id].numerical.equal(expected)


def test_flip_sign_handles_mixed_shared_and_canonical_groups() -> None:
    # Group 0: members 0 and 1 share a single stored row (like duplicate
    # `Choice` assignments).
    shared = TableTensor.from_tensor(torch.randn(1, 4, 3))
    solo = TableTensor.from_tensor(torch.randn(1, 4, 3))
    ensemble = EnsembleTable(
        groups=(shared, solo), locations=((0, 0), (0, 0), (1, 0))
    )
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(11),
    )

    assert len(output) == 3
    torch.testing.assert_close(
        output[0].numerical, shared.numerical[0] * processor._signs[0]
    )
    torch.testing.assert_close(
        output[1].numerical, shared.numerical[0] * processor._signs[1]
    )
    torch.testing.assert_close(
        output[2].numerical, solo.numerical[0] * processor._signs[2]
    )

    restored = processor.inverse_transform_ensemble(output)
    torch.testing.assert_close(restored[0].numerical, shared.numerical[0])
    torch.testing.assert_close(restored[1].numerical, shared.numerical[0])
    torch.testing.assert_close(restored[2].numerical, solo.numerical[0])


def test_flip_sign_falls_back_when_group_has_an_unreferenced_row() -> None:
    # Group 0 stores 2 rows but only member 0 (at position 0) references
    # it; row 1 is unreferenced.
    group0 = TableTensor.from_tensor(torch.randn(2, 4, 3))
    group1 = TableTensor.from_tensor(torch.randn(1, 4, 3))
    ensemble = EnsembleTable(
        groups=(group0, group1), locations=((0, 0), (1, 0))
    )
    processor = FlipSign()

    output = processor.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(13),
    )

    assert len(output) == 2
    torch.testing.assert_close(
        output[0].numerical, group0.numerical[0] * processor._signs[0]
    )
    torch.testing.assert_close(
        output[1].numerical, group1.numerical[0] * processor._signs[1]
    )
