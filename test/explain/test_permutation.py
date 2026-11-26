# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm
from sdm.explain import PermutationExplainer
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_permutation_explainer(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    x = sdm.TableTensor(
        columns={"numerical": ["A"], "categorical": ["B"]},
        numerical=torch.randn(32, 1, device=device),
        categorical=sdm.CategoricalTensor(
            code=torch.randint(0, 2, (32, 1), device=device),
            categories=(torch.arange(2, device=device),),
        ),
    )
    y = x["A"] if dtype.is_floating_point else x["B"]

    model = sdm.models.TabICLv2(pretrained=False, device=device)
    model.fit(x, y)

    out = PermutationExplainer(num_repeats=3).explain(model, x, y)
    assert out.size() == (3, 2)
    assert out.device == device
    assert out.numerical_columns == ("A", "B")
