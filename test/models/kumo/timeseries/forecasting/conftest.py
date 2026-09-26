# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def forecasting_rng() -> Iterator[None]:
    """Reproduce parity failures while preserving other tests' random state."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        yield
