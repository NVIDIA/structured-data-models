# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import FlipSign


def test_flip_sign() -> None:
    table = TableTensor.from_tensor(torch.randn(2, 5, 3))

    processor = FlipSign()
    out = processor.fit_transform(table)
    assert processor.sign.size() == (2, 1, 3)
    assert ((processor.sign == -1) | (processor.sign == 1)).all()
    assert out.numerical.equal(table.numerical * processor.sign)
    assert processor.inverse_transform(out).equal(table)
