# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import (
    CategoricalTensor,
    EnsembleData,
    EnsembleTable,
    EnsembleTensor,
    TableTensor,
)

pytestmark = pytest.mark.skipif(
    not hasattr(EnsembleData, "execute"),
    reason="Requires the ensemble execute implementation extracted from #797",
)


def test_replace_blocks_with_shared_tensor() -> None:
    table = TableTensor(
        columns={"numerical": ("value",), "categorical": ("category",)},
        numerical=torch.arange(3, dtype=torch.float32).view(3, 1),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0]]),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    ensemble = EnsembleTable.from_table(table, num_members=4)
    numerical = torch.full((3, 1), 7.0)

    output = ensemble.replace_blocks(numerical=numerical)

    for member in output:
        assert member.equal(table.replace_blocks(numerical=numerical))
        assert (
            member.categorical.categories[0] is table.categorical.categories[0]
        )
    assert ensemble[0].equal(table)


def test_replace_blocks_aligns_multiple_ensemble_inputs() -> None:
    first = TableTensor(
        numerical=torch.arange(12, dtype=torch.float32).view(2, 3, 2),
        datetime=torch.zeros(2, 3, 1, dtype=torch.long),
        columns={"numerical": ("first", "second"), "datetime": ("time",)},
    )
    second = TableTensor(
        numerical=torch.zeros(1, 4, 2),
        datetime=torch.zeros(1, 4, 1, dtype=torch.long),
        columns={"numerical": ("left", "right"), "datetime": ("timestamp",)},
    )
    ensemble = EnsembleTable(
        groups=(first, second),
        locations=((0, 1), (1, 0), (0, 0), (0, 1)),
    )
    numerical = EnsembleTensor(
        groups=(
            torch.full((1, 4, 2), 40.0),
            torch.arange(18, dtype=torch.float32).view(3, 3, 2),
        ),
        locations=((1, 2), (0, 0), (1, 0), (1, 1)),
    )
    datetime = EnsembleTensor(
        groups=(
            torch.full((1, 3, 1), 30, dtype=torch.long),
            torch.full((1, 4, 1), 40, dtype=torch.long),
        ),
        locations=((0, 0), (1, 0), (0, 0), (0, 0)),
    )

    output = ensemble.replace_blocks(numerical=numerical, datetime=datetime)

    for member_id in range(len(ensemble)):
        expected = ensemble[member_id].replace_blocks(
            numerical=numerical[member_id],
            datetime=datetime[member_id],
        )
        assert output[member_id].equal(expected)
        assert output[member_id].columns == ensemble[member_id].columns
