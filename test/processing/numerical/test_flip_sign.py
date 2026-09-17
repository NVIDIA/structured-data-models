# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import Choice, FlipSign, Identity


def test_flip_sign() -> None:
    table = TableTensor.from_tensor(torch.randn(2, 5, 3))

    processor = FlipSign()
    out = processor.fit_transform(table)
    assert processor.sign.size() == (2, 1, 3)
    assert ((processor.sign == -1) | (processor.sign == 1)).all()
    assert out.numerical.equal(table.numerical * processor.sign)
    assert processor.inverse_transform(out).equal(table)


def _signs(before: TableTensor, after: TableTensor) -> torch.Tensor:
    sign = after.numerical / before.numerical
    assert ((sign == -1) | (sign == 1)).all()
    assert (sign == sign[:1]).all()
    return sign[0]


def test_flip_sign_ensemble_draws_independent_member_signs() -> None:
    table = TableTensor.from_tensor(torch.randn(20, 64))
    ensemble = EnsembleTable.from_table(table, num_members=8)
    processor = FlipSign()

    output = processor.fit_transform_ensemble(ensemble)
    signs = torch.stack(
        [
            _signs(table, output.table(member_id))
            for member_id in range(output.num_members)
        ]
    )
    assert not (signs == signs[0]).all()

    restored = processor.inverse_transform_ensemble(output)
    for member_id in range(restored.num_members):
        assert restored.table(member_id).equal(table)

    query = TableTensor.from_tensor(torch.randn(5, 64))
    query_output = processor.transform_ensemble(
        EnsembleTable.from_table(query, num_members=8)
    )
    for member_id in range(query_output.num_members):
        assert query_output.table(member_id).numerical.equal(
            query.numerical * signs[member_id]
        )

    loaded = FlipSign()
    loaded.load_state_dict(processor.state_dict())
    loaded_output = loaded.transform_ensemble(ensemble)
    for member_id in range(loaded_output.num_members):
        assert loaded_output.table(member_id).equal(output.table(member_id))

    with pytest.raises(RuntimeError, match="ensemble members"):
        processor.transform_ensemble(
            EnsembleTable.from_table(table, num_members=4)
        )


def test_flip_sign_inside_choice() -> None:
    table = TableTensor.from_tensor(torch.randn(20, 64))
    ensemble = EnsembleTable.from_table(table, num_members=4)
    processor = Choice(FlipSign(), Identity(), method="round_robin")

    output = processor.fit_transform_ensemble(ensemble)
    for member_id in (1, 3):
        assert output.table(member_id).equal(table)
    for member_id in (0, 2):
        _signs(table, output.table(member_id))
