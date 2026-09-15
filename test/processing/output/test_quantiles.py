# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import EnsembleProcessor, ReduceQuantiles
from sdm.testing import withCUDA


@withCUDA
def test_reduce_quantiles(device: torch.device) -> None:
    data = torch.randn(2, 4, 3, device=device)
    table = TableTensor.from_tensor(data, columns=["q10", "q50", "q90"])

    output = ReduceQuantiles().transform(table)

    assert output.size() == (2, 4, 1)
    assert output.column_names == {"mean"}
    assert output.device == table.device
    assert output.dtype == data.dtype
    torch.testing.assert_close(
        output.numerical, data.mean(dim=-1, keepdim=True)
    )


@withCUDA
def test_reduce_quantiles_through_ensemble_adapter(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[0.0, 2.0, 4.0]], device=device)
    )
    second = TableTensor.from_tensor(
        torch.tensor([[1.0, 5.0, 9.0]], device=device)
    )
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    output = EnsembleProcessor.as_processor(
        ReduceQuantiles()
    ).transform_ensemble(ensemble_table)

    assert output.num_members == 2
    torch.testing.assert_close(
        output.table(0).numerical, torch.tensor([[2.0]], device=device)
    )
    torch.testing.assert_close(
        output.table(1).numerical, torch.tensor([[5.0]], device=device)
    )
    assert output.table(1).column_names == {"mean"}
