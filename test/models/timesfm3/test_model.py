# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from sdm import TableTensor
from sdm.models.timesfm3 import TimesFM3


def test_forward() -> None:
    model = TimesFM3(pretrained=False)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(5, 4),
        columns=["x1", "x2", "x3", "x4"],
    )
    x_query = TableTensor.from_tensor(
        torch.randn(3, 2),
        columns=["x2", "x4"],
    )

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(5, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(x_context, y_context, x_query)
    assert out.size() == (3, 3 * 9)
    assert "y1__q10" in out.columns["numerical"]
    assert "y2__q50" in out.columns["numerical"]
    assert "y3__q90" in out.columns["numerical"]

    model.fit(x_context, y_context)
    out = model.predict(x_query)
    assert out.allclose(model.predict(x_query))


@pytest.mark.parametrize("key", ["seqused_train", "seqused_cols"])
def test_seqused_rejected_before_preprocessing(key: str) -> None:
    model = TimesFM3(pretrained=False)
    x = torch.ones(5, 2)
    y = torch.ones(5, 1)
    query = torch.ones(3, 2)
    model.fit(x, y)
    expected = model.predict(query)
    kwargs: dict[str, Any] = {key: torch.tensor(2, dtype=torch.int32)}
    # The scalar cannot be converted to a table: padding rejection must
    # precede that conversion, and a rejected fit must preserve the cache.
    invalid = torch.tensor(1.0)
    with pytest.raises(ValueError, match="padding keywords"):
        model(invalid, y, query, **kwargs)
    with pytest.raises(ValueError, match="padding keywords"):
        model.fit(invalid, y, **kwargs)
    assert model.predict(query).allclose(expected)
