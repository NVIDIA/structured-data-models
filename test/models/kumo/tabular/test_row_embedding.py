# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.row_embedding import RowEmbedding
from sdm.testing import onlyCUDA, withCUDA


@withCUDA
def test_row_embedding(device: torch.device) -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=4,
        num_heads=2,
        group_size=3,
        num_frequencies=32,
        num_inducing_points=4,
        num_readout_tokens=2,
        device=device,
    )
    x = torch.randn(2, 5, 3, device=device)
    x[0, 0, 0] = torch.nan
    x[0, 4, 1] = torch.nan
    x[1, 1, 2] = torch.nan
    x[1, 3, 0] = torch.nan
    y = torch.randn(2, 3, device=device)
    categorical_mask = torch.zeros(2, 3, device=device, dtype=torch.bool)

    with torch.no_grad():
        expected = encoder(x, y, categorical_mask)
    assert expected.size() == (2, 5, 32)
    assert expected.device == device

    cache = Cache()
    with torch.no_grad():
        encoder(x[:, :3], y, categorical_mask, cache=cache)
        out = encoder(
            x[:, 3:],
            y[:, :0],
            categorical_mask,
            cache=cache.freeze(),
        )
    assert out.size() == (2, 2, 32)
    assert out.device == device
    torch.testing.assert_close(out, expected[:, 3:], atol=1e-5, rtol=1e-5)


@onlyCUDA
@pytest.mark.parametrize("cached", [False, True])
def test_row_embedding_passes(
    cached: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_frequencies=8,
        num_inducing_points=4,
        num_readout_tokens=2,
        device="cuda",
    )
    # Randomize the zero-initialized residual branches, so that attention
    # outputs reach the embedding.
    for parameter in encoder.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    # With many batch entries, e.g. estimators or ECOC tasks, a chunk of the
    # row attention spans more rows than the chunk memory limit allows.
    x = torch.randn(17, 64, 3, device="cuda")
    x[x > 1.0] = torch.nan
    y = torch.randn(17, 4, device="cuda")
    categorical_mask = torch.tensor([True, False, False], device="cuda")

    @torch.inference_mode()
    def embed() -> torch.Tensor:
        if not cached:
            return encoder(x, y, categorical_mask)
        cache = Cache()
        encoder(x[:, :4], y, categorical_mask, cache=cache)
        return encoder(
            x[:, 4:],
            y[:, :0],
            categorical_mask,
            cache=cache.freeze(),
        )

    expected = embed()
    # Small, intermediate and single-pass budgets exercise pass boundaries.
    total_memory = torch.cuda.get_device_properties("cuda").total_memory
    for budget in (2**10, 2**14, 2**19):
        fraction = budget / total_memory
        monkeypatch.setenv("SDM_CHUNK_MEMORY_FRACTION", str(fraction))
        torch.testing.assert_close(embed(), expected)


@onlyCUDA
def test_row_embedding_passes_on_wide_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Attention over more than 256 row tokens splits keys and values for
    # small batches of rows, so passes must keep the row chunks of one pass.
    encoder = RowEmbedding(
        num_classes=10,
        channels=64,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_frequencies=8,
        num_inducing_points=8,
        num_readout_tokens=2,
        device="cuda",
    )
    # Randomize the zero-initialized residual branches, so that attention
    # outputs reach the embedding.
    for parameter in encoder.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    x = torch.randn(3, 1169, 300, device="cuda")
    x[x > 2.0] = torch.nan
    y = torch.randint(10, (3, 200), device="cuda")
    categorical_mask = torch.arange(300, device="cuda") % 4 == 0

    def embed() -> tuple[torch.Tensor, int]:
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = encoder(x, y, categorical_mask)
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated() - baseline

    monkeypatch.setenv("SDM_CHUNK_MEMORY_FRACTION", "0.001")
    actual, peak = embed()
    monkeypatch.setattr(
        RowEmbedding,
        "_pass_starts",
        lambda self, x, train_size, cache: [0],
    )
    expected, expected_peak = embed()

    assert torch.equal(actual, expected)
    assert peak < expected_peak
