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


@withCUDA
def test_ecoc_members_draw_codebooks_in_order(device: torch.device) -> None:
    model = MyModel(10)
    ecoc = ECOC(max_classes=10)
    num_members, num_classes, num_context = 3, 12, 11
    x = torch.randn(
        num_members,
        num_context + 2,
        num_context,
        device=device,
        dtype=torch.float64,
    )
    y = torch.rand(num_members, num_classes, device=device).argsort(dim=-1)[
        ..., :num_context
    ]

    generator = torch.Generator(device).manual_seed(0)
    expected = torch.stack(
        [
            ecoc(
                model=model,
                x=x[member],
                y=y[member],
                num_classes=num_classes,
                generator=generator,
            )
            for member in range(num_members)
        ]
    )

    generator = torch.Generator(device).manual_seed(0)
    actual = ecoc(
        model=model,
        x=x,
        y=y,
        num_classes=num_classes,
        num_members=num_members,
        generator=generator,
    )
    torch.testing.assert_close(actual, expected)

    cache = Cache()
    ecoc(
        model=model,
        x=x[..., :num_context, :],
        y=y,
        num_classes=num_classes,
        num_members=num_members,
        cache=cache,
        generator=torch.Generator(device).manual_seed(0),
    )
    cache.freeze()
    replayed = ecoc(
        model=model,
        x=x[..., num_context:, :],
        y=y[..., :0],
        num_classes=num_classes,
        num_members=num_members,
        cache=cache,
    )
    torch.testing.assert_close(replayed, expected)


@pytest.mark.parametrize("num_classes", [3, 10, 11, 100, 201])
def test_ecoc_num_tasks(num_classes: int) -> None:
    ecoc = ECOC(max_classes=10)
    x = torch.randn(num_classes + 2, num_classes, dtype=torch.float64)
    y = torch.arange(num_classes)
    cache = Cache()

    ecoc(
        model=MyModel(10),
        x=x[:num_classes],
        y=y,
        num_classes=num_classes,
        cache=cache,
    )

    num_tasks = (
        cast(Tensor, cache["ecoc_codebook"]).size(-2)
        if num_classes > 10
        else 1
    )
    assert ecoc.num_tasks(num_classes) == num_tasks
