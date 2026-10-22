# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual Python examples in the NanoTabICL guide."""

import re
from pathlib import Path
from types import ModuleType

import pytest
import torch

from sdm.nn import Attention
from sdm.testing import withCUDA


@pytest.fixture(scope="module")
def guide() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "docs/source/nanotabicl.md"
    source = path.read_text(encoding="utf-8")
    blocks = list(
        re.finditer(r"^```python\n(.*?)^```\s*$", source, re.M | re.S)
    )
    assert blocks, "The guide must contain executable Python examples"
    module = ModuleType("nanotabicl_guide")
    for block in blocks:
        # Preserve document line numbers in tracebacks and share definitions
        # across snippets, just as a reader following the guide would.
        code = "\n" * source.count("\n", 0, block.start(1)) + block.group(1)
        exec(compile(code, str(path), "exec"), module.__dict__)
    return module


def _enable_attention(module: torch.nn.Module) -> None:
    # SDM initializes attention output projections to zero. Use identity
    # projections so isolation tests exercise information flow between rows.
    if isinstance(module, Attention):
        torch.nn.init.eye_(module.out_lin.weight)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("out_channels", [None, 1, 3])
def test_mlp(
    guide: ModuleType,
    device: torch.device,
    dtype: torch.dtype,
    out_channels: int | None,
) -> None:
    module = guide.mlp(8, out_channels=out_channels).to(
        device=device, dtype=dtype
    )
    x = torch.randn(2, 3, 8, device=device, dtype=dtype)
    with torch.inference_mode():
        out = module(x)

    assert out.shape == (2, 3, 8 if out_channels is None else out_channels)
    assert out.dtype == dtype
    assert out.device == device
    assert torch.isfinite(out).all()


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_ctx", [1, 3])
def test_column_transformer_context_isolation(
    guide: ModuleType,
    device: torch.device,
    dtype: torch.dtype,
    n_ctx: int,
) -> None:
    module = (
        guide.ColumnTransformer(
            channels=8,
            num_inducing_points=3,
            num_heads=2,
            num_layers=2,
        )
        .apply(_enable_attention)
        .to(device=device, dtype=dtype)
        .eval()
    )
    x = torch.randn(n_ctx + 2, 3, 8, device=device, dtype=dtype)
    original = x.clone()
    changed = x.clone()
    changed[-1] = 10 * changed[-1] + 5

    with torch.inference_mode():
        out = module(x, n_ctx)
        changed_out = module(changed, n_ctx)
        context_out = module(x[:n_ctx], n_ctx)
        separate_out = torch.cat(
            [
                module(torch.cat([x[:n_ctx], row.unsqueeze(0)]), n_ctx)[n_ctx:]
                for row in x[n_ctx:]
            ]
        )

    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device == device
    assert torch.isfinite(out).all()
    torch.testing.assert_close(x, original)
    torch.testing.assert_close(changed_out[:-1], out[:-1])
    torch.testing.assert_close(context_out, out[:n_ctx])
    torch.testing.assert_close(separate_out, out[n_ctx:])
    assert not torch.allclose(changed_out[-1], out[-1])


@withCUDA
def test_column_transformer_permutations(
    guide: ModuleType,
    device: torch.device,
) -> None:
    module = (
        guide.ColumnTransformer(
            channels=8,
            num_inducing_points=3,
            num_heads=2,
            num_layers=2,
        )
        .apply(_enable_attention)
        .to(device)
        .eval()
    )
    x = torch.randn(5, 3, 8, device=device)
    rows = torch.tensor([2, 0, 1, 4, 3], device=device)
    columns = torch.tensor([2, 0, 1], device=device)

    with torch.inference_mode():
        out = module(x, n_ctx=3)
        torch.testing.assert_close(module(x[rows], n_ctx=3), out[rows])
        torch.testing.assert_close(
            module(x[:, columns], n_ctx=3), out[:, columns]
        )


@withCUDA
@pytest.mark.parametrize("num_columns", [1, 5])
@pytest.mark.parametrize("num_readout_tokens", [1, 4])
def test_row_transformer_readout(
    guide: ModuleType,
    device: torch.device,
    num_columns: int,
    num_readout_tokens: int,
) -> None:
    module = (
        guide.RowTransformer(
            channels=8,
            num_layers=2,
            num_heads=2,
            num_readout_tokens=num_readout_tokens,
        )
        .apply(_enable_attention)
        .to(device)
        .eval()
    )
    x = torch.randn(3, num_columns, 8, device=device)

    with torch.inference_mode():
        out = module(x)
        separate_out = torch.cat([module(row.unsqueeze(0)) for row in x])
        permuted_out = module(x.flip(1))
        changed_out = module(x + 3)

    assert module.num_readout_tokens == num_readout_tokens
    assert out.shape == (3, num_readout_tokens, 8)
    assert out.device == device
    assert torch.isfinite(out).all()
    torch.testing.assert_close(separate_out, out)
    torch.testing.assert_close(permuted_out, out)
    assert not torch.allclose(changed_out, out)


@withCUDA
@pytest.mark.parametrize(("n_ctx", "n_query"), [(1, 1), (3, 2), (3, 0)])
def test_icl_transformer_context_only_attention(
    guide: ModuleType,
    device: torch.device,
    n_ctx: int,
    n_query: int,
) -> None:
    module = (
        guide.ICLTransformer(channels=8, num_heads=2, num_layers=2)
        .apply(_enable_attention)
        .to(device)
        .eval()
    )
    context = torch.randn(n_ctx, 8, device=device)
    query = torch.randn(n_query, 8, device=device)

    with torch.inference_mode():
        out = module(context, query)
        permuted_out = module(context.flip(0), query.flip(0))
        # An unrelated query must not influence any existing query, including
        # through context representations updated in intermediate layers.
        extra_query = torch.full((1, 8), 20.0, device=device)
        extended_out = module(context, torch.cat([query, extra_query]))
        changed_context_out = module(context + 3, query)
        for i in range(n_query):
            torch.testing.assert_close(
                module(context, query[i : i + 1]), out[i : i + 1]
            )

    assert out.shape == (n_query, 8)
    assert out.device == device
    assert torch.isfinite(out).all()
    torch.testing.assert_close(permuted_out, out.flip(0))
    torch.testing.assert_close(extended_out[:n_query], out)
    if n_query:
        assert not torch.allclose(changed_context_out, out)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("num_columns", "n_ctx", "n_query"), [(1, 1, 1), (5, 3, 2), (2, 3, 0)]
)
@pytest.mark.parametrize("custom", [False, True], ids=["defaults", "custom"])
def test_nanotabicl_logits(
    guide: ModuleType,
    device: torch.device,
    dtype: torch.dtype,
    num_columns: int,
    n_ctx: int,
    n_query: int,
    custom: bool,
) -> None:
    options = (
        {
            "num_readout_tokens": 2,
            "num_classes": 3,
            "num_heads": 2,
            "num_layers": 2,
            "num_inducing_points": 3,
        }
        if custom
        else {}
    )
    module = (
        guide.NanoTabICL(channels=8, **options)
        .to(device=device, dtype=dtype)
        .eval()
    )
    num_classes = 3 if custom else 2
    x = torch.randn(n_ctx + n_query, num_columns, device=device, dtype=dtype)
    labels = torch.arange(n_ctx, device=device) % num_classes
    original = x.clone()

    with torch.inference_mode():
        out = module(x, labels)

    assert out.shape == (n_query, num_classes)
    assert out.dtype == dtype
    assert out.device == device
    assert torch.isfinite(out).all()
    torch.testing.assert_close(x, original)


@pytest.fixture
def model(guide: ModuleType, device: torch.device) -> torch.nn.Module:
    return (
        guide.NanoTabICL(
            channels=8,
            num_readout_tokens=2,
            num_classes=3,
            num_heads=2,
            num_layers=2,
            num_inducing_points=3,
        )
        .apply(_enable_attention)
        .to(device)
        .eval()
    )


@withCUDA
def test_nanotabicl_permutations(
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    x = torch.randn(6, 4, device=device)
    labels = torch.tensor([0, 2, 1], device=device)
    context_order = torch.tensor([2, 0, 1], device=device)
    query_order = torch.tensor([2, 1, 0], device=device)
    column_order = torch.tensor([3, 0, 2, 1], device=device)

    with torch.inference_mode():
        out = model(x, labels)
        torch.testing.assert_close(
            model(
                torch.cat([x[:3][context_order], x[3:]]), labels[context_order]
            ),
            out,
        )
        torch.testing.assert_close(
            model(torch.cat([x[:3], x[3:][query_order]]), labels),
            out[query_order],
        )
        torch.testing.assert_close(model(x[:, column_order], labels), out)


@withCUDA
def test_nanotabicl_query_batch_independence(
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    x = torch.randn(6, 4, device=device)
    labels = torch.tensor([0, 2, 1], device=device)
    changed = x.clone()
    changed[4:] = 10 * changed[4:] + 5

    with torch.inference_mode():
        out = model(x, labels)
        separate_out = torch.cat(
            [
                model(torch.cat([x[:3], row.unsqueeze(0)]), labels)
                for row in x[3:]
            ]
        )
        changed_out = model(changed, labels)

    torch.testing.assert_close(separate_out, out)
    torch.testing.assert_close(changed_out[:1], out[:1])
    assert not torch.allclose(changed_out[1:], out[1:])


@withCUDA
def test_nanotabicl_uses_context_labels(
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    x = torch.randn(5, 3, device=device)
    labels = torch.zeros(3, dtype=torch.long, device=device)

    with torch.inference_mode():
        out = model(x, labels)
        relabeled_out = model(x, labels + 1)

    assert not torch.allclose(relabeled_out, out)


@pytest.mark.parametrize("label", [-1, 3])
def test_nanotabicl_rejects_out_of_range_labels(
    model: torch.nn.Module,
    label: int,
) -> None:
    # Keep invalid embedding indices on CPU to avoid a CUDA device assertion.
    with pytest.raises(IndexError):
        model(torch.ones(3, 2), torch.tensor([0, label]))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_columns", [1, 2, 7])
@pytest.mark.parametrize("shifts", [(0,), (0, 1, 3), (-1, 0, 9), (0, 0, 1)])
def test_feature_grouping_projection(
    guide: ModuleType,
    device: torch.device,
    dtype: torch.dtype,
    num_columns: int,
    shifts: tuple[int, ...],
) -> None:
    module = guide.FeatureGrouping(shift_pattern=shifts, embedding_dim=4).to(
        device=device, dtype=dtype
    )
    x = torch.arange(2 * num_columns, device=device, dtype=dtype).view(
        2, num_columns
    )
    with torch.no_grad():
        module.lin.weight.copy_(
            torch.arange(4 * len(shifts), device=device, dtype=dtype).view(
                4, -1
            )
            / 10
        )
        module.lin.bias.copy_(
            torch.tensor([0.5, -1.0, 2.0, -0.25], device=device, dtype=dtype)
        )

    # Independently index cyclic groups rather than repeating torch.roll.
    indices = (
        torch.arange(num_columns, device=device).unsqueeze(1)
        - torch.tensor(shifts, device=device)
    ) % num_columns
    expected = x[:, indices] @ module.lin.weight.T + module.lin.bias
    out = module(x)

    assert out.shape == (2, num_columns, 4)
    assert out.dtype == dtype
    assert out.device == device
    torch.testing.assert_close(out, expected)


def test_feature_grouping_defaults(guide: ModuleType) -> None:
    module = guide.FeatureGrouping()
    with torch.no_grad():
        module.lin.weight.zero_()
        module.lin.weight[:3].copy_(torch.eye(3))
        module.lin.bias.fill_(0.5)

    out = module(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    expected = torch.full((1, 4, 64), 0.5)
    expected[..., :3] += torch.tensor(
        [[[1.0, 4.0, 2.0], [2.0, 1.0, 3.0], [3.0, 2.0, 4.0], [4.0, 3.0, 1.0]]]
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_feature_grouping_requires_a_single_table(
    guide: ModuleType,
    shape: tuple[int, ...],
) -> None:
    module = guide.FeatureGrouping()
    with pytest.raises(AssertionError):
        module(torch.ones(shape))
