# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from sdm.nn import Attention
from sdm.testing import withCUDA


@pytest.fixture(scope="module")
def guide() -> dict[str, Any]:
    # Execute the published examples in order so tests cannot drift from them.
    path = Path(__file__).parents[2] / "docs/source/nanotabicl.md"
    namespace: dict[str, Any] = {}
    text = path.read_text()
    for match in re.finditer(r"^```python\n(.*?)^```", text, re.M | re.S):
        # Preserve guide line numbers in syntax errors and tracebacks.
        source = "\n" * text.count("\n", 0, match.start(1)) + match[1]
        exec(compile(source, str(path), "exec"), namespace)
    return namespace


def _initialize_attention(module: nn.Module) -> None:
    # SDM zero-initializes attention outputs. Exercise attention rather than
    # letting leakage and permutation tests pass through residuals alone.
    for child in module.modules():
        if isinstance(child, Attention):
            nn.init.eye_(child.out_lin.weight)
    module.eval()


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("out_channels", [None, 1, 3])
def test_mlp(
    guide: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    out_channels: int | None,
) -> None:
    module = guide["mlp"](8, out_channels=out_channels).to(
        device=device, dtype=dtype
    )
    x = torch.randn(2, 3, 8, device=device, dtype=dtype)

    out = module(x)

    assert out.shape == (2, 3, 8 if out_channels is None else out_channels)
    assert out.dtype == dtype
    assert out.device == device
    assert out.isfinite().all()


@pytest.mark.parametrize("num_layers", [1, 2])
@torch.no_grad()
def test_column_transformer_context_only_attention(
    guide: dict[str, Any], num_layers: int
) -> None:
    module = guide["ColumnTransformer"](
        channels=8,
        num_inducing_points=3,
        num_heads=2,
        num_layers=num_layers,
    )
    _initialize_attention(module)
    x = torch.randn(7, 3, 8)
    n_ctx = 4

    out = module(x, n_ctx)
    assert out.shape == x.shape

    changed_query = x.clone()
    changed_query[-1] += torch.randn_like(changed_query[-1]) * 5
    changed_out = module(changed_query, n_ctx)
    torch.testing.assert_close(changed_out[:-1], out[:-1])
    assert not torch.allclose(changed_out[-1], out[-1])

    # Neither the context nor a query may depend on companion queries.
    torch.testing.assert_close(module(x[:n_ctx], n_ctx), out[:n_ctx])
    for index in range(n_ctx, x.size(0)):
        single = torch.cat([x[:n_ctx], x[index : index + 1]])
        torch.testing.assert_close(module(single, n_ctx)[-1], out[index])

    changed_context = x.clone()
    changed_context[:n_ctx] += torch.randn_like(x[:n_ctx]) * 5
    assert not torch.allclose(
        module(changed_context, n_ctx)[n_ctx:], out[n_ctx:]
    )


@torch.no_grad()
def test_column_transformer_permutations(guide: dict[str, Any]) -> None:
    module = guide["ColumnTransformer"](
        channels=8, num_inducing_points=3, num_heads=2, num_layers=2
    )
    _initialize_attention(module)
    x = torch.randn(7, 3, 8)
    rows = torch.tensor([2, 0, 3, 1, 6, 4, 5])
    columns = torch.tensor([2, 0, 1])

    out = module(x, n_ctx=4)

    torch.testing.assert_close(module(x[rows], n_ctx=4), out[rows])
    torch.testing.assert_close(module(x[:, columns], n_ctx=4), out[:, columns])


@pytest.mark.parametrize("num_readout_tokens", [1, 4])
@torch.no_grad()
def test_row_transformer_readout(
    guide: dict[str, Any], num_readout_tokens: int
) -> None:
    module = guide["RowTransformer"](
        channels=8,
        num_layers=2,
        num_heads=2,
        num_readout_tokens=num_readout_tokens,
    )
    _initialize_attention(module)
    assert module.num_readout_tokens == num_readout_tokens

    # The same encoder produces a fixed readout width for different tables.
    for num_columns in (1, 5):
        x = torch.randn(3, num_columns, 8)
        out = module(x)

        assert out.shape == (3, num_readout_tokens, 8)
        torch.testing.assert_close(module(x.flip(1)), out)
        torch.testing.assert_close(module(x.flip(0)), out.flip(0))
        torch.testing.assert_close(module(x[:1]), out[:1])

        changed = x.clone()
        changed[0] += torch.randn_like(changed[0]) * 5
        changed_out = module(changed)
        torch.testing.assert_close(changed_out[1:], out[1:])
        assert not torch.allclose(changed_out[0], out[0])


@pytest.mark.parametrize("num_layers", [1, 2])
@torch.no_grad()
def test_icl_transformer_context_and_query_isolation(
    guide: dict[str, Any], num_layers: int
) -> None:
    module = guide["ICLTransformer"](
        channels=8, num_heads=2, num_layers=num_layers
    )
    _initialize_attention(module)
    context = torch.randn(4, 8)
    query = torch.randn(3, 8)
    out = module(context, query)

    assert out.shape == query.shape
    for index in range(query.size(0)):
        torch.testing.assert_close(
            module(context, query[index : index + 1]), out[index : index + 1]
        )
    torch.testing.assert_close(module(context.flip(0), query), out)
    torch.testing.assert_close(module(context, query.flip(0)), out.flip(0))

    changed_query = query.clone()
    changed_query[-1] += torch.randn_like(query[-1]) * 5
    changed_out = module(context, changed_query)
    torch.testing.assert_close(changed_out[:-1], out[:-1])
    assert not torch.allclose(changed_out[-1], out[-1])
    assert not torch.allclose(
        module(context + torch.randn_like(context), query), out
    )
    assert module(context, query[:0]).shape == (0, 8)


@pytest.fixture
def model(guide: dict[str, Any]) -> nn.Module:
    module = guide["NanoTabICL"](
        channels=8,
        num_readout_tokens=2,
        num_classes=3,
        num_heads=2,
        num_layers=2,
        num_inducing_points=3,
    )
    _initialize_attention(module)
    return module


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("num_context", "num_query", "num_columns", "num_classes"),
    [(1, 1, 1, 2), (4, 3, 5, 3), (4, 0, 2, 3)],
)
@torch.no_grad()
def test_nanotabicl_forward(
    guide: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    num_context: int,
    num_query: int,
    num_columns: int,
    num_classes: int,
) -> None:
    module = guide["NanoTabICL"](channels=8, num_classes=num_classes).to(
        device=device, dtype=dtype
    )
    _initialize_attention(module)
    x = torch.randn(
        num_context + num_query, num_columns, device=device, dtype=dtype
    )
    y = torch.arange(num_context, device=device) % num_classes

    out = module(x, y)

    assert out.shape == (num_query, num_classes)
    assert out.dtype == dtype
    assert out.device == device
    assert out.isfinite().all()
    probabilities = out.softmax(dim=-1)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(num_query, device=device, dtype=dtype),
    )


@torch.no_grad()
def test_nanotabicl_permutations(model: nn.Module) -> None:
    x = torch.randn(7, 5)
    y = torch.tensor([0, 1, 2, 1])
    context_order = torch.tensor([2, 0, 3, 1])
    query_order = torch.tensor([2, 0, 1])
    out = model(x, y)

    permuted_context = torch.cat([x[:4][context_order], x[4:]])
    torch.testing.assert_close(model(permuted_context, y[context_order]), out)
    permuted_query = torch.cat([x[:4], x[4:][query_order]])
    torch.testing.assert_close(model(permuted_query, y), out[query_order])
    torch.testing.assert_close(model(x.flip(1), y), out)


@torch.no_grad()
def test_nanotabicl_query_batch_independence(model: nn.Module) -> None:
    x = torch.randn(7, 5)
    y = torch.tensor([0, 1, 2, 1])
    out = model(x, y)

    separate = torch.cat(
        [model(torch.cat([x[:4], row.unsqueeze(0)]), y) for row in x[4:]]
    )
    torch.testing.assert_close(separate, out)

    changed = x.clone()
    changed[-1] += torch.randn_like(changed[-1]) * 5
    changed_out = model(changed, y)
    torch.testing.assert_close(changed_out[:-1], out[:-1])
    assert not torch.allclose(changed_out[-1], out[-1])


@pytest.mark.parametrize("label_stage", ["column", "icl"])
@torch.no_grad()
def test_nanotabicl_uses_context_labels(
    model: nn.Module, label_stage: str
) -> None:
    # Isolate each documented label injection so dropping either is caught.
    if label_stage == "column":
        model.icl_label_embedding.weight.zero_()
    else:
        model.column_label_embedding.weight.zero_()
    x = torch.randn(7, 5)
    y = torch.zeros(4, dtype=torch.long)

    assert not torch.allclose(model(x, y), model(x, y + 1))


@pytest.mark.parametrize("label", [-1, 3])
def test_nanotabicl_invalid_context_label(
    model: nn.Module, label: int
) -> None:
    with pytest.raises(IndexError):
        model(torch.randn(3, 2), torch.tensor([0, label]))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("num_columns", "shift_pattern"),
    [
        (7, (0, 1, 3)),
        (2, (0, 1, 3)),
        (1, (0, 1, 3)),
        (4, (0,)),
        (4, (-1, 0, 6)),
    ],
)
@torch.no_grad()
def test_feature_grouping_offsets(
    guide: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    num_columns: int,
    shift_pattern: tuple[int, ...],
) -> None:
    module = guide["FeatureGrouping"](
        shift_pattern=shift_pattern, embedding_dim=len(shift_pattern)
    ).to(device=device, dtype=dtype)
    module.lin.weight.copy_(
        torch.eye(len(shift_pattern), device=device, dtype=dtype)
    )
    module.lin.bias.zero_()
    x = torch.arange(2 * num_columns, device=device, dtype=dtype).reshape(
        2, num_columns
    )

    out = module(x)

    assert out.shape == (2, num_columns, len(shift_pattern))
    assert out.dtype == dtype
    assert out.device == device
    # An independent indexing oracle checks shift direction and wraparound.
    for column in range(num_columns):
        expected = x[
            :, [(column - shift) % num_columns for shift in shift_pattern]
        ]
        torch.testing.assert_close(out[:, column], expected)


@torch.no_grad()
def test_feature_grouping_projection(guide: dict[str, Any]) -> None:
    module = guide["FeatureGrouping"](embedding_dim=2)
    module.lin.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
    module.lin.bias.copy_(torch.tensor([0.5, -2.0]))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    out = module(x)

    expected = torch.tensor(
        [
            [[15.5, -1.0], [13.5, -1.0], [19.5, -1.0], [13.5, -5.0]],
            [[39.5, -1.0], [37.5, -1.0], [43.5, -1.0], [37.5, -5.0]],
        ]
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("shape", [(3,), (2, 3, 4)])
def test_feature_grouping_rejects_non_matrix(
    guide: dict[str, Any], shape: tuple[int, ...]
) -> None:
    module = guide["FeatureGrouping"]()

    with pytest.raises(AssertionError):
        module(torch.zeros(shape))
