# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

import sdm
from sdm.explain import ShapleyExplainer
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("imputer", ["marginal", "baseline"])
def test_shapley_explainer(
    imputer: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    pytest.importorskip("shapiq")

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

    output = "q500" if dtype.is_floating_point else "0"
    prediction = model.predict(x[:2])[output].numerical.squeeze(-1).cpu()

    out = ShapleyExplainer(imputer=imputer).explain(
        model,
        x[:2],
        output=output,
        background=x[:4],
        budget=4,
    )
    assert len(out) == 2
    for value, expected in zip(out, prediction):
        attributions = value.get_n_order_values(1)
        assert attributions.shape == (2,)
        assert np.isfinite(attributions).all()
        np.testing.assert_allclose(
            attributions.sum() + value.baseline_value,
            expected.numpy(),
            atol=1e-5,
        )
