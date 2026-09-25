# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import cast

import pytest
import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.models import ECOC
from sdm.testing import withCUDA


class MyModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        temperature: float = 1.0,
        cache: Cache | None = None,
    ) -> Tensor:
        probs = (x[..., y.size(-1) :, :] / temperature).softmax(dim=-1)
        if cache is not None:
            y = cast(Tensor, cache.setdefault("y", y))
        out = probs.new_zeros((*probs.shape[:-1], self.num_classes))
        return out.scatter_add(
            dim=-1,
            index=y.unsqueeze(-2).expand_as(probs),
            src=probs,
        ).log()


@withCUDA
@pytest.mark.parametrize(
    ("max_classes", "num_classes", "batch_shape"),
    [(10, 3, ()), (10, 11, ()), (10, 201, (2, 3)), (2, 12, ())],
)
def test_ecoc(
    device: torch.device,
    max_classes: int,
    num_classes: int,
    batch_shape: tuple[int, ...],
) -> None:
    model = MyModel(max_classes)
    ecoc = ECOC(max_classes=max_classes)
    forward = partial(
        ecoc, model=model, num_classes=num_classes, temperature=0.7
    )
    # Scramble the labels and leave one class absent from each context.
    num_context = num_classes - 1
    x = torch.randn(
        *batch_shape,
        num_context + 3,
        num_context,
        device=device,
        dtype=torch.float64,
    )
    y = torch.rand(*batch_shape, num_classes, device=device).argsort(dim=-1)[
        ..., :num_context
    ]

    scores = forward(x=x, y=y)
    probs = (x[..., -3:, :] / 0.7).softmax(dim=-1)
    expected = x.new_zeros((*batch_shape, 3, num_classes)).scatter(
        dim=-1,
        index=y.unsqueeze(-2).expand_as(probs),
        src=probs,
    )

    torch.testing.assert_close(scores.exp(), expected)

    cache = Cache()
    recorded = forward(x=x[..., :num_context, :], y=y, cache=cache)
    assert recorded.shape == (*batch_shape, 0, num_classes)
    cache.freeze()
    for query, expected in zip(
        x[..., num_context:, :].split(2, dim=-2),
        scores.split(2, dim=-2),
        strict=True,
    ):
        actual = forward(x=query, y=y[..., :0], cache=cache)
        torch.testing.assert_close(actual, expected)

    if num_classes > ecoc.max_classes:
        with pytest.raises(ValueError, match=r"num_classes.*cached"):
            forward(
                x=x[..., -1:, :],
                y=y[..., :0],
                num_classes=num_classes + 1,
                cache=cache,
            )
