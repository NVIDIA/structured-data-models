# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.testing import withCUDA


class _StubICLBlock(ICLBlock):
    def __init__(
        self,
        num_classes: int,
        predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            out_channels=max(num_classes, 1),
            channels=4,
            num_layers=0,
            num_heads=1,
            norm_bias=True,
        )
        self.predictor = predictor

    def _forward(self, x: Tensor, y: Tensor, **_: object) -> Tensor:
        return self.predictor(x, y)


def test_icl_block_grouping() -> None:
    block = _StubICLBlock(
        num_classes=10,
        predictor=lambda x, y: x,
    )
    device = torch.device("cpu")

    assignments, num_groups = block._grouping(
        num_classes=5,
        device=device,
    )
    assert num_groups == 1
    torch.testing.assert_close(
        assignments,
        torch.zeros(5, dtype=torch.long, device=device),
    )

    assignments, num_groups = block._grouping(
        num_classes=25,
        device=device,
    )
    assert num_groups == 3
    torch.testing.assert_close(
        assignments,
        torch.tensor([0] * 9 + [1] * 8 + [2] * 8, device=device),
    )

    assignments, num_groups = block._grouping(
        num_classes=101,
        device=device,
    )
    assert num_groups == 10
    torch.testing.assert_close(
        assignments,
        torch.arange(10, device=device).repeat_interleave(
            torch.tensor([11] + [10] * 9, device=device)
        ),
    )


@withCUDA
@pytest.mark.parametrize(
    (
        "total_num_classes",
        "num_classes",
        "expected_calls",
        "expected_probabilities",
    ),
    [
        pytest.param(
            3,
            2,
            2,
            [1 / 4, 1 / 4, 1 / 2],
            id="single-level-with-singleton",
        ),
        pytest.param(
            5,
            2,
            4,
            [1 / 8, 1 / 8, 1 / 4, 1 / 4, 1 / 4],
            id="multi-level-with-singleton",
        ),
    ],
)
def test_icl_block_hierarchical_log_probs(
    device: torch.device,
    total_num_classes: int,
    num_classes: int,
    expected_calls: int,
    expected_probabilities: list[float],
) -> None:
    calls = 0

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        test_size = rows.size(0) - labels.size(0)
        return rows.new_zeros((test_size, num_classes))

    test_size = 2
    row_embeddings = torch.randn(
        total_num_classes + test_size,
        4,
        device=device,
    )
    y = torch.arange(total_num_classes, device=device)
    block = _StubICLBlock(
        num_classes=num_classes,
        predictor=predictor,
    )

    log_probs = block(
        row_embeddings,
        y,
        num_classes=total_num_classes,
    )

    assert calls == expected_calls
    torch.testing.assert_close(
        log_probs,
        row_embeddings.new_tensor(expected_probabilities)
        .log()
        .expand(test_size, -1),
    )


def test_icl_block_hierarchical_nonuniform_log_probs() -> None:
    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        test_size = rows.size(0) - labels.size(0)
        if labels.size(0) == 3:
            probabilities = rows.new_tensor([0.6, 0.4])
        else:
            probabilities = rows.new_tensor([0.25, 0.75])
        return probabilities.log().expand(test_size, -1)

    block = _StubICLBlock(num_classes=2, predictor=predictor)
    row_embeddings = torch.randn(5, 4)
    log_probs = block(
        row_embeddings,
        torch.arange(3),
        num_classes=3,
    )

    expected = torch.tensor([0.15, 0.45, 0.4]).log().expand(2, -1)
    torch.testing.assert_close(log_probs, expected)


def test_icl_block_hierarchical_batched_absent_classes() -> None:
    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique(sorted=True)
        torch.testing.assert_close(
            classes,
            torch.arange(classes.numel(), device=labels.device),
        )
        test_size = rows.size(0) - labels.size(0)
        return rows.new_zeros((test_size, 2))

    block = _StubICLBlock(num_classes=2, predictor=predictor)
    row_embeddings = torch.randn(2, 5, 4)
    y = torch.tensor([[1, 3, 4], [0, 2, 5]])

    log_probs = block(
        row_embeddings,
        y,
        num_classes=6,
    )

    expected_probabilities = torch.tensor(
        [
            [0.0, 0.25, 0.0, 0.25, 0.5, 0.0],
            [0.25, 0.0, 0.25, 0.0, 0.0, 0.5],
        ],
    )
    torch.testing.assert_close(
        log_probs,
        expected_probabilities.log().unsqueeze(1).expand(-1, 2, -1),
    )


def test_icl_block_hierarchical_preserves_gradients() -> None:
    logits = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return logits + rows[labels.size(0) :, :2]

    block = _StubICLBlock(
        num_classes=2,
        predictor=predictor,
    )
    row_embeddings = torch.randn(5, 4, requires_grad=True)
    log_probs = block(
        row_embeddings,
        torch.arange(3),
        num_classes=3,
    )
    log_probs[:, 0].sum().backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    assert row_embeddings.grad is not None
    assert row_embeddings.grad.abs().sum() > 0


def test_icl_block_hierarchical_empty_batch() -> None:
    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise AssertionError("The predictor must not run for an empty batch")

    block = _StubICLBlock(num_classes=2, predictor=predictor)
    row_embeddings = torch.randn(0, 4, 3)
    log_probs = block(
        row_embeddings,
        torch.zeros(0, 2, dtype=torch.long),
        num_classes=5,
    )

    assert log_probs.size() == (0, 2, 5)
    assert log_probs.dtype == torch.float32


def test_icl_block_rejects_one_class_hierarchy() -> None:
    block = _StubICLBlock(
        num_classes=1,
        predictor=lambda x, y: x,
    )

    with pytest.raises(
        ValueError,
        match="Hierarchical classification requires 'num_classes'",
    ):
        block(
            torch.randn(3, 4),
            torch.tensor([0, 1]),
            num_classes=2,
        )


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_icl_block_hierarchical_cache(
    device: torch.device,
    batch_shape: tuple[int, ...],
    requires_grad: bool,
) -> None:
    block = ICLBlock(
        num_classes=2,
        out_channels=2,
        channels=4,
        num_layers=2,
        num_heads=2,
        norm_bias=True,
        temperature=0.9,
        device=device,
    )
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.1)

    num_classes, test_size = 5, 2
    train_rows = torch.randn(*batch_shape, num_classes, 4, device=device)
    test_rows = torch.randn(*batch_shape, test_size, 4, device=device)
    y = torch.arange(num_classes, device=device).expand(
        *batch_shape,
        num_classes,
    )

    expected = block(
        torch.cat((train_rows, test_rows), dim=-2),
        y,
        num_classes=num_classes,
    )

    with torch.set_grad_enabled(requires_grad):
        cache = Cache()
        recorded = block(
            train_rows.clone(),
            y,
            num_classes=num_classes,
            cache=cache,
        )
        assert recorded.size() == (*batch_shape, 0, num_classes)
        assert cache.size() > 0

        predicted = block(
            test_rows.clone(),
            y.new_empty((*batch_shape, 0)),
            num_classes=num_classes,
            cache=cache.freeze(),
        )
    torch.testing.assert_close(predicted, expected)
