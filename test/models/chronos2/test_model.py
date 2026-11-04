# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Any

import pytest
import torch

from sdm import TableTensor
from sdm.models import Chronos2
from sdm.models.chronos2 import model as chronos2


@pytest.mark.parametrize("cross_learning", [False, True])
def test_forward(cross_learning: bool) -> None:
    model = Chronos2(pretrained=False)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(2, 5, 4),
        columns=["x1", "x2", "x3", "x4"],
    )
    x_query = TableTensor.from_tensor(
        torch.randn(2, 3, 2),
        columns=["x2", "x4"],
    )

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(2, 5, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(
        x_context=x_context,
        y_context=y_context,
        x_query=x_query,
        num_estimators=1,
        cross_learning=cross_learning,
    )
    assert out.size() == (2, 3, 3 * 21)
    assert out.numerical.isfinite().all()
    assert "y1__q10" in out.numerical_columns
    assert "y2__q50" in out.numerical_columns
    assert "y3__q90" in out.numerical_columns

    model.fit(
        x=x_context,
        y=y_context,
        num_estimators=1,
        context_only_columns=["x1", "x3"],
        cross_learning=cross_learning,
    )
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)


@pytest.mark.parametrize("key", ["seqused_train", "seqused_cols"])
def test_seqused_rejected_before_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    monkeypatch.setattr(
        target=chronos2,
        name="_Chronos2",
        value=partial(
            chronos2._Chronos2,
            channels=8,
            hidden_channels=16,
            num_layers=1,
            num_heads=2,
            patch_size=4,
            context_length=16,
        ),
    )
    model = Chronos2(pretrained=False)
    x = torch.ones(5, 2)
    y = torch.ones(5, 1)
    query = torch.ones(3, 2)
    model.fit(x, y)
    expected = model.predict(query)
    kwargs: dict[str, Any] = {key: torch.tensor(2, dtype=torch.int32)}

    # Reject padding before slicing inputs and preserve the fitted cache.
    for context in (x, torch.tensor(1.0)):
        with pytest.raises(ValueError, match="padding keywords"):
            model(context, y, query, **kwargs)
        with pytest.raises(ValueError, match="padding keywords"):
            model.fit(context, y, **kwargs)
        assert model.predict(query).allclose(expected)
