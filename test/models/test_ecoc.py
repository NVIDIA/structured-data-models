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
        return (
            probabilities.new_zeros(
                (*probabilities.shape[:-1], self.num_classes)
            )
            .scatter_add(
                dim=-1,
                index=y.unsqueeze(-2).expand(*probabilities.shape),
                src=probabilities,
            )
            .log()
        )


@withCUDA
@pytest.mark.parametrize("num_classes", [1, 3, 10])
def test_ecoc_within_capacity(device: torch.device, num_classes: int) -> None:
    model = _ProbabilityClassifier(10).to(device)
    ecoc = ECOC(model, max_classes=10)
    x = torch.randn(2, num_classes + 3, num_classes, device=device)
    y = torch.arange(num_classes, device=device).expand(2, -1)

    actual = ecoc(x, y, num_classes=num_classes, temperature=0.7)
    expected = model(x, y, temperature=0.7)[..., :num_classes]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@withCUDA
@pytest.mark.parametrize("num_classes", [11, 23, 201])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_ecoc_recovers_probabilities(
    device: torch.device,
    num_classes: int,
    batch_shape: tuple[int, ...],
) -> None:
    ecoc = ECOC(_ProbabilityClassifier(10), max_classes=10).to(device)
    x = torch.randn(*batch_shape, num_classes + 3, num_classes, device=device)
    y = torch.rand(*batch_shape, num_classes, device=device).argsort(dim=-1)

    scores = ecoc(x, y, num_classes=num_classes, temperature=0.7)
    probabilities = (x[..., -3:, :] / 0.7).softmax(dim=-1)
    expected = torch.empty_like(probabilities).scatter(
        dim=-1,
        index=y.unsqueeze(-2).expand_as(probabilities),
        src=probabilities,
    )

    torch.testing.assert_close(scores.exp(), expected)
    torch.testing.assert_close(scores.softmax(dim=-1), expected)
    assert scores.device == device
    assert scores.dtype == x.dtype


@pytest.mark.parametrize("max_classes", [2, 10])
def test_ecoc_absent_classes(max_classes: int) -> None:
    ecoc = ECOC(_ProbabilityClassifier(max_classes), max_classes=max_classes)
    x = torch.randn(7, 4, dtype=torch.float64)
    y = torch.tensor([0, 3, 7, 11])

    scores = ecoc(x, y, num_classes=12)
    expected = x.new_zeros(3, 12).scatter(
        dim=-1,
        index=y.expand(3, -1),
        src=x[-3:].softmax(dim=-1),
    )

    torch.testing.assert_close(scores.softmax(dim=-1), expected)


@pytest.mark.parametrize("num_classes", [3, 17])
def test_ecoc_cache(num_classes: int) -> None:
    ecoc = ECOC(_ProbabilityClassifier(10), max_classes=10)
    x = torch.randn(2, num_classes + 5, num_classes)
    y = torch.arange(num_classes).expand(2, -1)
    expected = ecoc(x, y, num_classes=num_classes)
    cache = Cache()

    recorded = ecoc(
        x[..., :num_classes, :], y, num_classes=num_classes, cache=cache
    )
    cache = cache.freeze().to("cpu")
    actual = torch.cat(
        [
            ecoc(query, y[..., :0], num_classes=num_classes, cache=cache)
            for query in x[..., num_classes:, :].split(2, dim=-2)
        ],
        dim=-2,
    )

    assert recorded.shape == (2, 0, num_classes)
    torch.testing.assert_close(actual, expected)


def test_ecoc_cache_class_count() -> None:
    ecoc = ECOC(_ProbabilityClassifier(10), max_classes=10)
    x = torch.randn(17, 17)
    y = torch.arange(17)
    cache = Cache()
    ecoc(x, y, num_classes=17, cache=cache)

    with pytest.raises(
        ValueError, match="must match the cached ECOC codebook"
    ):
        ecoc(x[:1], y[:0], num_classes=18, cache=cache.freeze())


def test_ecoc_gradients() -> None:
    model = _ProbabilityClassifier(10)
    ecoc = ECOC(model, max_classes=10)
    x = torch.randn(15, 12, requires_grad=True)
    y = torch.arange(12)

    scores = ecoc(x, y, num_classes=12)
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

    ecoc = ECOC(Classifier(), max_classes=10)
    x = torch.randn(15, 12)
    y = torch.arange(12)

    first = ecoc(
        x, y, num_classes=12, generator=torch.Generator().manual_seed(1)
    )
    repeated = ecoc(
        x, y, num_classes=12, generator=torch.Generator().manual_seed(1)
    )
    other = ecoc(
        x, y, num_classes=12, generator=torch.Generator().manual_seed(2)
    )

    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.allclose(first, other)
