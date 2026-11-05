# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.kumo.timeseries.anomaly.threshold import (
    macs_threshold,
    scs_threshold,
)
from sdm.testing import withCUDA


@withCUDA
def test_threshold(device: torch.device) -> None:
    scores = torch.full((2, 400), 0.1, device=device)
    scores[0, [0, 50, 200, 350]] = 2
    target = torch.ones(2, 400, 1, device=device)

    expected = torch.zeros_like(scores, dtype=torch.bool)
    expected[0, [50, 350]] = True
    torch.testing.assert_close(scs_threshold(scores, target), expected)

    expected[0, 200] = True
    torch.testing.assert_close(macs_threshold(scores), expected)
