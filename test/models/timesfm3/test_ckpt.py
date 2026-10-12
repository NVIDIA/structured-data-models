# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.models.timesfm3 import TimesFM3
from sdm.models.timesfm3 import model as model_module
from sdm.models.timesfm3.model import _TimesFM3
from sdm.testing import withCUDA


def _checkpoint() -> dict[str, Tensor]:
    shapes = {
        "pre_transformer_resblock.hidden_layer.weight": (8, 12),
        "pre_transformer_resblock.output_layer.weight": (8, 8),
        "pre_transformer_resblock.residual_layer.weight": (8, 12),
        "output_head.weight": (36, 8),
        "output_head.bias": (36,),
    }
    for layer in range(2):
        prefix = f"transformer_stack.layers.{layer}."
        for name in (
            "pre_seq_attn_ln",
            "post_seq_attn_ln",
            "pre_var_attn_ln",
            "post_var_attn_ln",
            "pre_ff_ln",
            "post_ff_ln",
        ):
            shapes[prefix + name + ".weight"] = (8,)
        for name in ("ff0", "ff1"):
            shapes[prefix + name + ".weight"] = (8, 8)
        for side in ("seq", "var"):
            attn = prefix + side + "_attn."
            for name in ("query", "key", "value", "out"):
                shapes[attn + name + "_proj.weight"] = (8, 8)
            for name in ("query", "key"):
                shapes[attn + name + "_ln.weight"] = (4,)
            shapes[attn + "per_dim_scale.per_dim_scale"] = (4,)
    out = {}
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        value = (torch.arange(math.prod(shape)).float() + i).sin()
        if name.endswith("_ln.weight"):
            value = 1 + 0.05 * value
        else:
            value = 0.1 * value
        out[name] = value.reshape(shape)
    return out


@withCUDA
def test_load_checkpoint_matches_upstream(
    device: torch.device,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.safetensors"
    save_file(_checkpoint(), path)
    monkeypatch.setattr(
        model_module,
        "download_checkpoint",
        lambda **kwargs: str(path),
    )
    monkeypatch.setattr(
        model_module,
        "_TimesFM3",
        partial(
            _TimesFM3,
            input_patch_size=2,
            output_patch_size=4,
            channels=8,
            num_layers=2,
            num_heads=2,
        ),
    )
    model = TimesFM3(pretrained=True, device=device)
    known = torch.tensor([[0.2], [1.0], [0.5], [2.0]], device=device)
    past = torch.tensor([[1.0], [2.0], [3.0], [2.0]], device=device)
    context = TableTensor.from_tensor(
        torch.cat([known, past], dim=-1),
        columns=["known", "past"],
    )
    query = TableTensor.from_tensor(
        torch.tensor([[1.5], [2.5], [0.8], [3.0], [1.0]], device=device),
        columns=["known"],
    )
    target = TableTensor.from_tensor(
        torch.tensor(
            [[0.0, 0.5], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], device=device
        ),
        columns=["a", "b"],
    )
    reference = json.loads(
        Path(__file__).with_name("checkpoint_reference.json").read_text()
    )
    expected = torch.tensor(reference["forecast"], device=device).flatten(-2)
    out = model(context, target, query)
    assert out.columns[Stype.numerical] == tuple(
        f"{name}__q{i}" for name in ("a", "b") for i in range(10, 100, 10)
    )
    torch.testing.assert_close(out.numerical, expected, rtol=1e-5, atol=1e-5)
    model.fit(x=context, y=target, context_only_columns=["past"])
    torch.testing.assert_close(
        model.predict(query).numerical, expected, rtol=1e-5, atol=1e-5
    )


@withCUDA
def test_univariate_horizon_rounding(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module,
        "_TimesFM3",
        partial(
            _TimesFM3,
            input_patch_size=2,
            output_patch_size=4,
            channels=8,
            num_layers=2,
            num_heads=2,
        ),
    )
    model = TimesFM3(pretrained=False, device=device)
    model.model.load_state_dict(model_module.remap_ckpt(_checkpoint()))
    context = torch.empty(4, 0, device=device)
    target = torch.tensor([[0.0], [1.0], [2.0], [3.0]], device=device)
    query = torch.empty(8, 0, device=device)
    # Requests rounded to the same output-patch boundary share a forecast.
    expected = model(context, target, query).numerical[:5]
    actual = model(context, target, query[:5]).numerical
    torch.testing.assert_close(actual, expected)
