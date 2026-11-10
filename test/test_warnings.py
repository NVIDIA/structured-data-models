# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm._warnings import warn_once


def test_warn_once_compiles(compilation: None) -> None:
    def add_one(x: torch.Tensor) -> torch.Tensor:
        warn_once(key="test-warn-once-compiles", message="Unused.")
        return x + 1

    compiled = torch.compile(add_one, backend="eager", fullgraph=True)
    x = torch.zeros(2)
    torch.testing.assert_close(compiled(x), x + 1)
