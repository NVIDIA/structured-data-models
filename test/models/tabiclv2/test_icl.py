# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
import torch

from sdm.cache import Cache
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.testing import withCUDA


@pytest.fixture(autouse=True)
def reset_compiler() -> Iterator[None]:
    torch.compiler.reset()
    # Use strict FP32 when comparing batched and compact matrix operations.
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


def _block(device: torch.device, num_classes: int = 2) -> ICLBlock:
    block = ICLBlock(
        num_classes=num_classes,
        out_channels=num_classes,
        channels=8,
        num_layers=2,
        num_heads=2,
        norm_bias=True,
        temperature=0.9,
        device=device,
    ).eval()
    # Exercise attention and its context-length scaling: residual projections
    # are zero-initialized by default and would hide incorrect masking.
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    for module in block.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    return block


@withCUDA
@pytest.mark.parametrize(
    ("capacity", "probabilities"),
    [
        (2, [1 / 4, 1 / 4, 1 / 2]),
        (2, [1 / 8, 1 / 8, 1 / 4, 1 / 4, 1 / 4]),
        (10, [1 / 27] * 9 + [1 / 24] * 16),
        (10, [1 / 120] * 6 + [1 / 100] * 5 + [1 / 100] * 90),
    ],
)
def test_icl_block_hierarchical_log_probs(
    device: torch.device,
    capacity: int,
    probabilities: list[float],
) -> None:
    block = _block(device=device, num_classes=capacity)
    for parameter in block.head.parameters():
        torch.nn.init.zeros_(parameter)
    num_classes = len(probabilities)
    x = torch.randn(num_classes + 2, 8, device=device)
    y = torch.arange(num_classes, device=device)
    expected = (
        x.new_tensor(probabilities).log().expand(2, -1) * block.temperature
    )
    with torch.no_grad():
        torch.testing.assert_close(
            block(x, y, num_classes=num_classes), expected
        )
        compiled = torch.compile(block, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x, y, num_classes=num_classes), expected
        )


@withCUDA
@pytest.mark.parametrize("requires_grad", [False, True])
def test_icl_block_hierarchical_compile(
    device: torch.device, requires_grad: bool
) -> None:
    block = _block(device=device)
    compiled = torch.compile(block, backend="eager", fullgraph=True)
    # Two tables have different absent classes and class frequencies. Swapping
    # labels changes which context rows belong to each node.
    labels = torch.tensor(
        [[1, 3, 4, 4, 4, 4, 4, 4, 4], [5, 0, 2, 5, 5, 5, 5, 5, 5]],
        device=device,
    )
    with torch.set_grad_enabled(requires_grad):
        for y in (labels, labels.flip(-1)):
            x = torch.randn(
                2, 11, 8, device=device, requires_grad=requires_grad
            )
            actual = compiled(x, y, num_classes=6)
            expected = []
            for rows, targets in zip(x, y):
                classes, local_y = targets.unique(
                    sorted=True, return_inverse=True
                )
                # Explicit three-class reference: a two-class root routes to
                # a two-class leaf and a singleton with probability one.
                root = block(rows.clone(), local_y // 2, num_classes=2)
                root = (root / block.temperature).log_softmax(dim=-1)
                selected = local_y < 2
                leaf_rows = torch.cat((rows[:-2][selected], rows[-2:]))
                leaf = block(leaf_rows, local_y[selected], num_classes=2)
                leaf = (leaf / block.temperature).log_softmax(dim=-1)
                probs = torch.cat((root[:, :1] + leaf, root[:, 1:]), dim=-1)
                expected.append(
                    probs.new_full((2, 6), -torch.inf).index_copy(
                        -1, classes, probs
                    )
                )
            reference = torch.stack(expected) * block.temperature
            torch.testing.assert_close(actual, reference, atol=2e-6, rtol=1e-5)
            if requires_grad:
                inputs = (x, *block.parameters())
                actual_grads = torch.autograd.grad(
                    actual[actual.isfinite()].sum(), inputs
                )
                reference_grads = torch.autograd.grad(
                    reference[reference.isfinite()].sum(), inputs
                )
                for actual_grad, reference_grad in zip(
                    actual_grads, reference_grads
                ):
                    torch.testing.assert_close(
                        actual_grad, reference_grad, atol=2e-6, rtol=1e-4
                    )
                assert actual_grads[0].abs().sum() > 0


@withCUDA
@pytest.mark.parametrize("labels", [[0, 1, 0], [4, 4, 4], [1, 3, 5]])
def test_icl_block_hierarchical_absent_classes(
    device: torch.device, labels: list[int]
) -> None:
    block = _block(device=device)
    for parameter in block.head.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(5, 8, device=device)
    y = torch.tensor(labels, device=device, dtype=torch.int32)
    present = sorted(set(labels))
    probabilities = (
        [0.25, 0.25, 0.5]
        if len(present) == 3
        else [1 / len(present)] * len(present)
    )
    expected = x.new_zeros(6)
    expected[present] = x.new_tensor(probabilities)
    with torch.no_grad():
        compiled = torch.compile(block, backend="eager", fullgraph=True)
        actual = compiled(x, y, num_classes=6)
    torch.testing.assert_close(
        actual, expected.log().expand(2, -1) * block.temperature
    )


@withCUDA
def test_tabiclv2_hierarchical_compile(device: torch.device) -> None:
    model = _TabICLv2(
        num_classes=2,
        num_quantiles=3,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=2,
        group_size=1,
        num_readout_tokens=1,
        num_icl_layers=1,
        num_icl_heads=2,
        device=device,
    ).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    x = torch.randn(1, 5, 2, device=device)
    y = torch.tensor([[0, 1, 2]], device=device)
    with torch.no_grad():
        expected = model(x, y, num_classes=3)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x, y, num_classes=3), expected)
        cache = Cache()
        recorded = compiled(x[:, :3].clone(), y, num_classes=3, cache=cache)
        assert recorded.size() == (1, 0, 3)
        actual = compiled(
            x[:, 3:].clone(), y[:, :0], num_classes=3, cache=cache.freeze()
        )
        torch.testing.assert_close(actual, expected)


@withCUDA
def test_icl_block_hierarchical_autocast(device: torch.device) -> None:
    block = _block(device=device)
    compiled = torch.compile(block, backend="eager", fullgraph=True)
    x = torch.randn(6, 8, device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for labels in ([0, 1, 2, 3], [4, 4, 4, 4]):
            y = torch.tensor(labels, device=device)
            expected = block(x, y, num_classes=5)
            actual = compiled(x, y, num_classes=5)
            # Eager autocast may change dtype when singleton nodes appear.
            assert actual.dtype == torch.float32
            torch.testing.assert_close(
                actual, expected, atol=0.025, rtol=0.02, check_dtype=False
            )


def test_icl_block_hierarchical_empty_batch() -> None:
    block = _block(device=torch.device("cpu"))
    out = block(
        torch.randn(0, 4, 8),
        torch.zeros(0, 2, dtype=torch.long),
        num_classes=5,
    )
    assert out.size() == (0, 2, 5)
    assert out.dtype == torch.float32


@withCUDA
def test_icl_block_hierarchical_changing_labels(device: torch.device) -> None:
    block = _block(device=device)
    for parameter in block.head.parameters():
        torch.nn.init.zeros_(parameter)
    compiled = torch.compile(block, backend="eager", fullgraph=True)
    x = torch.randn(7, 8, device=device)
    cases = [
        ([0, 1, 2, 3, 4], [0.125, 0.125, 0.25, 0.25, 0.25]),
        ([4, 4, 4, 4, 4], [0, 0, 0, 0, 1]),
        ([1, 3, 1, 3, 1], [0, 0.5, 0, 0.5, 0]),
        ([4, 0, 2, 0, 4], [0.25, 0, 0.25, 0, 0.5]),
    ]
    with torch.no_grad():
        for labels, probabilities in cases:
            actual = compiled(
                x, torch.tensor(labels, device=device), num_classes=5
            )
            expected = x.new_tensor(probabilities).log() * block.temperature
            torch.testing.assert_close(actual, expected.expand(2, -1))


@withCUDA
@pytest.mark.parametrize(
    ("batch_size", "num_classes", "message"),
    [
        (1, 5, "Cached hierarchy batch shape"),
        (2, None, "Cached hierarchy class count"),
        (2, 2, "Cached hierarchy class count"),
        (2, 4, "Cached hierarchy class count"),
        (2, 6, "Cached hierarchy class count"),
    ],
)
def test_icl_block_hierarchical_cache_shape(
    device: torch.device,
    batch_size: int,
    num_classes: int | None,
    message: str,
) -> None:
    block = _block(device=device)
    compiled = torch.compile(block, backend="eager", fullgraph=True)
    cache = Cache()
    with torch.no_grad():
        compiled(
            torch.randn(2, 5, 8, device=device),
            torch.arange(5, device=device).expand(2, -1),
            num_classes=5,
            cache=cache,
        )
        with pytest.raises(RuntimeError, match=message):
            block(
                torch.randn(batch_size, 2, 8, device=device),
                torch.empty(batch_size, 0, dtype=torch.long, device=device),
                num_classes=num_classes,
                cache=cache.freeze(),
            )


def test_icl_block_rejects_one_class_hierarchy() -> None:
    block = _block(device=torch.device("cpu"), num_classes=1)
    with pytest.raises(
        ValueError, match="Hierarchical classification requires 'num_classes'"
    ):
        block(torch.randn(3, 8), torch.tensor([0, 1]), num_classes=2)


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_icl_block_hierarchical_cache(
    device: torch.device,
    batch_shape: tuple[int, ...],
    requires_grad: bool,
    compiled: bool,
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
        torch.nn.init.normal_(parameter, std=0.3)

    num_classes, test_size = 5, 2
    train_rows = torch.randn(*batch_shape, num_classes, 4, device=device)
    test_rows = torch.randn(*batch_shape, test_size, 4, device=device)
    train_rows.requires_grad_(requires_grad)
    test_rows.requires_grad_(requires_grad)
    y = torch.arange(num_classes, device=device).expand(
        *batch_shape,
        num_classes,
    )

    with torch.set_grad_enabled(requires_grad):
        expected = block(
            torch.cat((train_rows, test_rows), dim=-2),
            y,
            num_classes=num_classes,
        )
        predict = (
            torch.compile(block, backend="eager", fullgraph=True)
            if compiled
            else block
        )
        cache = Cache()
        recorded = predict(
            train_rows.clone(),
            y,
            num_classes=num_classes,
            cache=cache,
        )
        assert recorded.size() == (*batch_shape, 0, num_classes)
        assert cache.size() > 0

        predicted = predict(
            test_rows.clone(),
            y.new_empty((*batch_shape, 0)),
            num_classes=num_classes,
            cache=cache.freeze(),
        )
    torch.testing.assert_close(predicted, expected)
    if requires_grad:
        inputs = (train_rows, test_rows, *block.parameters())
        actual_grads = torch.autograd.grad(predicted.sum(), inputs)
        expected_grads = torch.autograd.grad(expected.sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, atol=2e-6, rtol=1e-4
            )
