# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import Asinh


def test_asinh_transform_and_inverse() -> None:
    table = TableTensor.from_tensor(torch.randn(3, 2))
    processor = Asinh()

    out = processor.transform(table)
    torch.testing.assert_close(out.numerical, table.numerical.asinh())
    out = processor.inverse_transform(out)
    torch.testing.assert_close(out.numerical, table.numerical)
