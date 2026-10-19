# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from functools import partial
from importlib.metadata import distribution
from typing import Any, Literal

import pytest
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors.torch import load_file
from torch import Tensor

from sdm.models.timestral import TimestralTransformerBlock
from sdm.nn import RotaryEmbedding, TransformerBlock
from sdm.testing import withCUDA

SOURCE_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
WEIGHT_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


@withCUDA
@pytest.mark.parametrize("gate", [False, True])
def test_causal_attention(device: torch.device, gate: bool) -> None:
    block = TimestralTransformerBlock(
        channels=16,
        num_heads=2,
        hidden_channels=24,
        attn_bias=True,
        mlp_bias=True,
        gate=gate,
        device=device,
    )
    # Shared attention starts with a zero output projection. Exercise its
    # contribution when checking that future tokens cannot affect the past.
    for module in block.modules():
        if isinstance(module, TransformerBlock):
            torch.nn.init.normal_(module.attn.out_lin.weight)
    x = torch.randn(2, 6, 16, device=device)
    changed = x.clone()
    changed[:, 3:] += 10
    actual = block(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual[:, :3], block(changed)[:, :3])


@pytest.fixture(scope="module")
def weights() -> dict[str, Tensor]:
    pytest.importorskip("timestral.layers")
    installed = json.loads(
        distribution("timestral").read_text("direct_url.json") or "{}"
    )
    if installed.get("vcs_info", {}).get("commit_id") != SOURCE_REVISION:
        pytest.skip("Parity requires the pinned upstream Timestral revision")
    try:
        path = hf_hub_download(
            "mistralai/Timestral-5M",
            "model.safetensors",
            revision=WEIGHT_REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip("Parity requires cached Timestral-5M weights")
    return load_file(path)


def load_reference(reference: Any, weights: dict[str, Tensor]) -> None:
    layer = (
        0 if len(reference.attns) == 1 and reference.attns[0].is_time else 3
    )
    prefix = f"transformer.layers.{layer}."
    state = {}
    for key in reference.state_dict():
        parts = key.split(".")
        if parts[0] in {"attns", "attn_norms"}:
            axis = int(parts[1])
            parts[1] = "0" if reference.attns[axis].is_time else "1"
        state[key] = weights[prefix + ".".join(parts)]
    reference.load_state_dict(state, strict=True)


def load_block(block: TimestralTransformerBlock, reference: Any) -> None:
    # Upstream partial RoPE pairs channels across full head halves. SDM
    # rotates a contiguous prefix, so permute Q/K rows but leave V unchanged.
    permutation = torch.tensor(
        [*range(16), *range(32, 48), *range(16, 32), *range(48, 64)],
        device=next(reference.parameters()).device,
    )
    state = {
        k: v for k, v in block.state_dict().items() if k.endswith(".inv_freq")
    }
    for index, attn in enumerate(reference.attns):
        prefix = f"layers.{index}."
        qkv = attn.wQKV.weight.reshape(3, 64, 3, 192).transpose(1, 2)
        if attn.is_time:
            qkv = torch.cat([qkv[:2].index_select(2, permutation), qkv[2:]])
        state[prefix + "attn.qkv_lin.weight"] = qkv.reshape(576, 192)
        state[prefix + "attn.out_lin.weight"] = attn.wO.weight
        state[prefix + "attn.gate.0.weight"] = attn._gate.weight
        state[prefix + "query_norm.weight"] = reference.attn_norms[index].scale
    prefix = f"layers.{len(reference.attns) - 1}.mlp."
    state[prefix + "0.weight"] = reference.mlp_norm.scale
    up, gate = reference.mlp.w12.weight.chunk(2, dim=0)
    state[prefix + "1.up_lin.weight"] = up
    state[prefix + "1.gate_lin.weight"] = gate
    state[prefix + "1.down_lin.weight"] = reference.mlp.w3.weight
    block.load_state_dict(state, strict=True)


@withCUDA
@pytest.mark.parametrize(
    "kind", ["time", "variate", "time_variate", "variate_time"]
)
@pytest.mark.parametrize("masked", [False, True])
def test_pretrained_parity(
    device: torch.device,
    kind: Literal["time", "variate", "time_variate", "variate_time"],
    masked: bool,
    weights: dict[str, Tensor],
) -> None:
    upstream = pytest.importorskip("timestral.layers")
    reference = upstream.TransformerBlock(
        kind=kind,
        embed_dim=192,
        num_heads=3,
        mlp_hidden_dim=512,
        norm_eps=1e-6,
        attn_bias=False,
        mlp_bias=False,
        time_attn_gate="sdpa_sigmoid",
        variate_attn_gate="sdpa_sigmoid",
    )
    load_reference(reference, weights)
    rope = RotaryEmbedding(
        channels=64,
        layout="split_half",
        theta=10000,
        partial_rotary_factor=0.5,
        requires_grad=False,
    )
    block = TimestralTransformerBlock(192, 3, 512, kind=kind, rope=rope)
    load_block(block, reference)
    reference, block = reference.to(device), block.to(device)
    ref_rope = upstream.SinCosRoPE(dim=64, base=10000, rotary_dim=32).to(
        device
    )
    # Independent series with three variates and repeated stream positions.
    x = (
        (torch.arange(6 * 8 * 192, device=device).float() / 53)
        .sin()
        .reshape(6, 8, 192)
    )
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=device).expand(6, -1)
    time_mask = None
    if masked:
        time_mask = torch.ones(8, 8, dtype=torch.bool, device=device).tril()
        time_mask[:, 1] = False
        time_mask.fill_diagonal_(True)
        time_mask = time_mask.unsqueeze(0)
    groups = torch.arange(6, device=device) // 3
    variate_mask = (groups[:, None] == groups[None, :]).unsqueeze(0)
    with torch.inference_mode():
        cos, sin = ref_rope(x, pos)
        expected = reference(x, time_mask, variate_mask, cos, sin)
        actual = block(x, time_mask, variate_mask, pos)
    # Synthetic inputs yield activations up to 1e3 with cancellation near
    # zero. Compare relative to activation RMS; Q/K permutation changes
    # floating-point reduction order.
    scale = expected.square().mean().sqrt().clamp_min(1)
    torch.testing.assert_close(
        actual / scale, expected / scale, atol=1e-5, rtol=1e-5
    )


@withCUDA
@pytest.mark.parametrize("known_covariates", [False, True])
def test_forecast_activations(
    device: torch.device,
    known_covariates: bool,
    weights: dict[str, Tensor],
) -> None:
    upstream = pytest.importorskip("timestral")
    forecaster = upstream.TimestralForecaster.from_pretrained(
        "mistralai/Timestral-5M",
        revision=WEIGHT_REVISION,
        device=device,
        local_files_only=True,
    )
    captured = {}

    def capture(index: int, module: torch.nn.Module, args: tuple) -> None:
        if index not in captured:
            captured[index] = args

    handles = [
        forecaster.model.transformer.layers[index].register_forward_pre_hook(
            partial(capture, index)
        )
        for index in (0, 3)
    ]
    t = torch.arange(160, device=device).float()
    target = (t[:96] / 9).sin().expand(2, 3, 96).clone()
    target += torch.arange(3, device=device)[None, :, None] * 0.2
    known = (t / 7).cos().expand(2, 1, 160) if known_covariates else None
    try:
        forecaster.forecast(target, 64, known_covariates=known)
    finally:
        for handle in handles:
            handle.remove()
    for index, kind in ((0, "time"), (3, "time_variate")):
        reference = forecaster.model.transformer.layers[index]
        rope = RotaryEmbedding(
            channels=64,
            layout="split_half",
            theta=10000,
            partial_rotary_factor=0.5,
            requires_grad=False,
            device=device,
        )
        block = TimestralTransformerBlock(
            192, 3, 512, kind=kind, rope=rope, device=device
        )
        load_block(block, reference)
        x, time_mask, variate_mask, cos, sin = captured[index]
        positions = torch.arange(
            x.size(1) // 2 if known_covariates else x.size(1), device=device
        )
        if known_covariates:
            positions = positions.repeat(2)
        positions = positions.expand(x.size(0), -1)
        full_mask = (
            time_mask if isinstance(time_mask, Tensor) else time_mask.mask
        )
        with torch.inference_mode():
            expected = reference(x, time_mask, variate_mask, cos, sin)
            actual = block(x, full_mask, variate_mask, positions)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
