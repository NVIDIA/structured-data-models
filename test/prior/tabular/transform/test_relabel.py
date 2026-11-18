# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.functions import RandomFunction
from sdm.prior.tabular.transform import Relabel


class _FirstFeature(RandomFunction):
    def forward(
        self,
        x: torch.Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        assert out_channels == 1
        return x[:, :1]


def test_relabel_recomputes_regression_and_classification_targets() -> None:
    values = torch.arange(4, dtype=torch.float32).repeat_interleave(2)
    x = TableTensor.from_tensor(values[:, None])
    transform = Relabel(_FirstFeature(), exact_probability=1.0)

    regression = TableTensor.from_tensor(torch.randn(8, 1))
    output_x, output = transform(x, regression)
    assert output_x.numerical.equal(x.numerical)
    assert torch.isclose(output.numerical.mean(), torch.tensor(0.0))
    assert torch.isclose(output.numerical.std(correction=1), torch.tensor(1.0))

    classification = TableTensor(
        categorical=CategoricalTensor(
            torch.zeros(8, 1, dtype=torch.long),
            categories=(torch.arange(3),),
        ),
    )
    _, output = transform(
        x=x, y=classification, generator=torch.Generator().manual_seed(1)
    )
    _, replay = transform(
        x=x, y=classification, generator=torch.Generator().manual_seed(1)
    )
    labels = output.categorical.code[:, 0]
    assert labels.equal(replay.categorical.code[:, 0])
    assert labels.unique().numel() == 3
    assert all(
        labels[values == value].unique().numel() == 1
        for value in values.unique()
    )
