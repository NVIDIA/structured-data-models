# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import Clip
from sdm.testing import withCUDA


@withCUDA
def test_clip_clamps_fixed_bounds(
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [[-101.0, -100.0], [100.0, 101.0]],
            device=device,
        ),
    )

    actual = Clip(-100.0, 100.0).transform(table)

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor(
            [[-100.0, -100.0], [100.0, 100.0]],
            device=device,
        ),
    )
    assert repr(Clip(-100.0, 100.0)) == "Clip(-100.0, 100.0)"


def test_clip_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="min_value"):
        Clip(1.0, -1.0)
