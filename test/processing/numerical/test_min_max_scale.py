# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import MinMaxScale
from sdm.testing import withCUDA


@withCUDA
def test_min_max_scale(device: torch.device) -> None:
    data = torch.tensor([[1.0, 5.0], [3.0, 5.0]], device=device)
    processor = MinMaxScale()
    out = processor.fit_transform(TableTensor.from_tensor(data))
    torch.testing.assert_close(
        out.numerical, data.new_tensor([[0.0, 0.0], [1.0, 0.0]])
    )

    query = data.new_tensor([[-1.0, 3.0], [5.0, 7.0]])
    out = processor.transform(TableTensor.from_tensor(query))
    torch.testing.assert_close(
        out.numerical, query.new_tensor([[-1.0, -2.0], [2.0, 2.0]])
    )
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical, query
    )
