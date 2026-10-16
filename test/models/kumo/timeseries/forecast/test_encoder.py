# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.kumo.timeseries.forecast.encoder import T5Encoder
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("length", [9, 140])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("num_buckets", "max_distance"), [(4, 2), (32, 128)])
def test_encoder_matches_t5(
    device: torch.device,
    length: int,
    dtype: torch.dtype,
    num_buckets: int,
    max_distance: int,
) -> None:
    transformers = pytest.importorskip("transformers")
    reference = transformers.T5EncoderModel(
        transformers.T5Config(
            vocab_size=16,
            d_model=32,
            d_ff=48,
            num_layers=2,
            num_heads=4,
            d_kv=8,
            dropout_rate=0.0,
            feed_forward_proj="gated-gelu",
            relative_attention_num_buckets=num_buckets,
            relative_attention_max_distance=max_distance,
        )
    ).eval()
    torch.nn.Module.to(reference, device=device, dtype=dtype)
    model = T5Encoder(
        channels=32,
        hidden_channels=48,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        dtype=dtype,
        device=device,
        num_buckets=num_buckets,
        max_distance=max_distance,
    ).eval()
    model.relative_attention_bias.load_state_dict(
        reference.encoder.block[0]
        .layer[0]
        .SelfAttention.relative_attention_bias.state_dict()
    )
    model.norm.load_state_dict(reference.encoder.final_layer_norm.state_dict())
    for i, block in enumerate(reference.encoder.block):
        model.get_submodule(f"layers.{i}.attention_norm").load_state_dict(
            block.layer[0].layer_norm.state_dict()
        )
        model.get_submodule(f"layers.{i}.ffn_norm").load_state_dict(
            block.layer[1].layer_norm.state_dict()
        )
        for name in ["q", "k", "v", "o"]:
            model.get_submodule(f"layers.{i}.{name}").load_state_dict(
                getattr(block.layer[0].SelfAttention, name).state_dict()
            )
        for name in ["wi_0", "wi_1", "wo"]:
            model.get_submodule(f"layers.{i}.{name}").load_state_dict(
                getattr(block.layer[1].DenseReluDense, name).state_dict()
            )
    x = torch.randn(2, length, 32, dtype=dtype, device=device)
    mask = torch.ones(2, length, dtype=torch.bool, device=device)
    mask[0, :3] = False
    mask[1, -5:] = False

    with torch.inference_mode():
        expected = reference(
            inputs_embeds=x, attention_mask=mask
        ).last_hidden_state
        actual = model(x, mask)
    assert actual.dtype == dtype
    if dtype == torch.bfloat16:
        # Fused SDPA/GELU avoid intermediate bf16 rounding in the reference.
        # Bound aggregate numerical error rather than near-zero elements.
        relative_error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm()
        assert relative_error < 0.02
    else:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_masked_keys_cannot_change_observed_outputs() -> None:
    model = T5Encoder(
        channels=16,
        hidden_channels=24,
        num_layers=2,
        num_heads=2,
    ).eval()
    x = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, :4] = False
    changed = x.clone()
    changed[:, :4] += 100
    with torch.inference_mode():
        torch.testing.assert_close(
            model(x, mask)[:, 4:], model(changed, mask)[:, 4:]
        )


def test_encoder_backward() -> None:
    model = T5Encoder(
        channels=16,
        hidden_channels=24,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    x = torch.randn(2, 12, 16, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_masked_keys_cannot_change_other_outputs(
    dtype: torch.dtype,
) -> None:
    model = T5Encoder(
        channels=16,
        hidden_channels=24,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        dtype=dtype,
    )
    x = torch.randn(2, 12, 16, dtype=dtype, requires_grad=True)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    changed = x.detach().clone()
    changed[:, -1] += torch.arange(16, dtype=dtype)

    actual = model(x, mask)
    with torch.no_grad():
        other = model(changed, mask)
    assert actual.isfinite().all()
    # Residual/FFN paths may change the final patch, but attention must not
    # carry it to any other patch when every key is masked.
    torch.testing.assert_close(actual[:, :-1], other[:, :-1], atol=0, rtol=0)
    actual.float().square().sum().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()


@pytest.mark.parametrize(
    ("num_buckets", "max_distance"), [(2, 128), (32, 8), (32, 7)]
)
def test_invalid_relative_position_configuration(
    num_buckets: int, max_distance: int
) -> None:
    with pytest.raises(ValueError, match=r"num_buckets.*max_distance"):
        T5Encoder(
            channels=16,
            hidden_channels=24,
            num_layers=1,
            num_heads=2,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )
