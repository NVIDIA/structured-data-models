# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The row executor rejects padding without changing native padded tables."""

from unittest.mock import patch

import pytest
import torch
from test.execution.test_tabular import fixture
from test.execution.test_tabular import group as group

from sdm import TableTensor, Task
from sdm.cache import Cache
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_rows import Rows


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("owner_local_frontend", [False, True])
def test_padded_columns_are_native_only(group, task, owner_local_frontend):
    model, x, target, _ = fixture(task)
    labels = (
        target.categorical.code.squeeze(-1)
        if isinstance(target, TableTensor)
        else target.squeeze(-1)
    )
    encoder = model.models[Task(task)].row_embedding
    mask = torch.zeros(x.size(-1), dtype=torch.bool)
    valid = x.size(-1) - 1

    with torch.no_grad():
        native = encoder(x, labels, mask, cache=Cache())
        rows = Rows(
            encoder,
            x,
            labels,
            mask,
            Exchange(group, group),
            8 << 30,
            owner_local_frontend=owner_local_frontend,
        )
        sharded = encoder(x, labels, mask, cache=Cache(), _runner=rows)
        torch.testing.assert_close(sharded, native, atol=2e-6, rtol=1e-5)

        with (
            patch.object(
                torch,
                "empty",
                side_effect=AssertionError("allocated before padding guard"),
            ),
            pytest.raises(
                ValueError,
                match="does not support padded columns",
            ),
        ):
            encoder(
                x,
                labels,
                mask,
                cache=Cache(),
                num_columns=torch.tensor(valid),
                _runner=rows,
            )

        padded = encoder(
            x.unsqueeze(0),
            labels.unsqueeze(0),
            mask.unsqueeze(0),
            num_columns=torch.tensor([valid]),
        )
        clipped = encoder(
            x[None, :, :valid],
            labels.unsqueeze(0),
            mask[None, :valid],
        )
        torch.testing.assert_close(padded, clipped, atol=1e-4, rtol=1e-4)
