# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from sdm.models.timesfm3.core import _TimesFM3Model
from sdm.testing import withCUDA


def _residual_config(output_dims: int = 8) -> dict[str, Any]:
    return {
        "hidden_dims": 8,
        "output_dims": output_dims,
        "use_bias": False,
    }


def _transformer_config(model_dims: int = 8) -> dict[str, Any]:
    return {
        "num_layers": 2,
        "transformer": {
            "model_dims": model_dims,
            "hidden_dims": 12,
            "num_heads": 2,
            "qk_norm": "rms",
            "use_rope_seq": True,
            "use_rope_var": False,
            "use_bias": False,
        },
    }


def _internal_model(
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    *,
    use_iterative_cpm_revin: bool = True,
    use_linear_detrending: bool = True,
) -> _TimesFM3Model:
    return _TimesFM3Model(
        input_patch_len=2,
        output_patch_len=4,
        quantiles=[0.1, 0.5, 0.9],
        residual_block_config=_residual_config(),
        transformer_config=_transformer_config(),
        use_iterative_cpm_revin=use_iterative_cpm_revin,
        use_linear_detrending=use_linear_detrending,
        device=device,
        dtype=dtype,
    )


@withCUDA
def test_internal_model_assembles_checkpoint_parameters(
    device: torch.device,
) -> None:
    source = _internal_model(device)
    state = source.state_dict()
    assert state["pre_transformer_resblock.hidden_layer.weight"].shape == (
        8,
        12,
    )
    assert state[
        "transformer_stack.layers.0.seq_attn.qkv_lin.weight"
    ].shape == (
        24,
        8,
    )
    assert state["output_head.weight"].shape == (12, 8)

    loaded = _internal_model("meta")
    loaded.load_state_dict(state, strict=True, assign=True)
    assert all(parameter.device == device for parameter in loaded.parameters())
    torch.testing.assert_close(
        loaded.output_head.weight, source.output_head.weight
    )
