# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.models import ECOC
from sdm.testing import withCUDA


class _ProbabilityClassifier(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        temperature: float = 1.0,
        cache: Cache | None = None,
    ) -> Tensor:
        # Each feature supplies the probability of its context row's class.
        # Sum probabilities for rows assigned the same encoded label.
        query = x[..., y.size(-1) :, :]
        if cache is not None:
            if cache.is_recording:
                cache["labels"] = y
            else:
                y = cast(Tensor, cache["labels"])
        probabilities = (query * self.scale / temperature).softmax(dim=-1)
        out = probabilities.new_zeros(
            (*probabilities.shape[:-1], self.num_classes)
        )
        return out.scatter_add(
            dim=-1,
            index=y.unsqueeze(-2).expand_as(probabilities),
            src=probabilities,
        ).log()


@pytest.mark.parametrize("num_classes", [3, 10])
def test_ecoc_within_capacity(num_classes: int) -> None:
    model = _ProbabilityClassifier(10)
    ecoc = ECOC(max_classes=10)
    x = torch.randn(2, num_classes + 3, num_classes)
    y = torch.arange(num_classes).expand(2, -1)

    actual = ecoc(model, x, y, num_classes=num_classes, temperature=0.7)
    expected = model(x, y, temperature=0.7)[..., :num_classes]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@withCUDA
@pytest.mark.parametrize(
    ("max_classes", "num_classes", "batch_shape"),
    [(10, 11, ()), (10, 201, (2, 3)), (2, 12, ())],
)
def test_ecoc_recovers_probabilities(
    device: torch.device,
    max_classes: int,
    num_classes: int,
    batch_shape: tuple[int, ...],
) -> None:
    model = _ProbabilityClassifier(max_classes).to(device)
    ecoc = ECOC(max_classes=max_classes)
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

    scores = ecoc(model, x, y, num_classes=num_classes, temperature=0.7)
    probabilities = (x[..., -3:, :] / 0.7).softmax(dim=-1)
    expected = x.new_zeros((*batch_shape, 3, num_classes)).scatter(
        dim=-1,
        index=y.unsqueeze(-2).expand_as(probabilities),
        src=probabilities,
    )

    torch.testing.assert_close(scores.exp(), expected)


@pytest.mark.parametrize("num_classes", [3, 17])
def test_ecoc_cache(num_classes: int) -> None:
    model = _ProbabilityClassifier(10)
    ecoc = ECOC(max_classes=10)
    x = torch.randn(2, num_classes + 5, num_classes)
    y = torch.arange(num_classes).expand(2, -1)
    expected = ecoc(model, x, y, num_classes=num_classes)
    cache = Cache()

    recorded = ecoc(
        model=model,
        x=x[..., :num_classes, :],
        y=y,
        num_classes=num_classes,
        cache=cache,
    )
    cache.freeze()
    actual = torch.cat(
        [
            ecoc(
                model, query, y[..., :0], num_classes=num_classes, cache=cache
            )
            for query in x[..., num_classes:, :].split(2, dim=-2)
        ],
        dim=-2,
    )

    assert recorded.shape == (2, 0, num_classes)
    torch.testing.assert_close(actual, expected)

    if num_classes > ecoc.max_classes:
        with pytest.raises(ValueError, match=r"num_classes.*cached"):
            ecoc(
                model=model,
                x=x[..., -1:, :],
                y=y[..., :0],
                num_classes=num_classes + 1,
                cache=cache,
            )


def test_ecoc_gradients() -> None:
    model = _ProbabilityClassifier(10)
    ecoc = ECOC(max_classes=10)
    x = torch.randn(15, 12, requires_grad=True)
    y = torch.arange(12)

    scores = ecoc(model, x, y, num_classes=12)
    loss = -scores[:, 0].mean()
    grad_x, grad_scale = torch.autograd.grad(loss, (x, model.scale))
    expected = -(x[-3:] * model.scale).log_softmax(dim=-1)[:, 0].mean()
    expected_x, expected_scale = torch.autograd.grad(
        expected, (x, model.scale)
    )

    torch.testing.assert_close(grad_x, expected_x)
    torch.testing.assert_close(grad_scale, expected_scale)


def test_ecoc_generator() -> None:
    # The oracle's output is invariant to the codebook. A fixed linear head
    # instead makes codebook differences observable in the decoded scores.
    class Classifier(torch.nn.Module):
        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x[..., y.size(-1) :, :10]

    model = Classifier()
    ecoc = ECOC(max_classes=10)
    x = torch.randn(15, 12)
    y = torch.arange(12)

    first, repeated, other = [
        ecoc(
            model=model,
            x=x,
            y=y,
            num_classes=12,
            generator=torch.Generator().manual_seed(seed),
        )
        for seed in (1, 1, 2)
    ]

    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.allclose(first, other)
