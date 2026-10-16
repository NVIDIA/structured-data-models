# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch import Tensor


def model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Read architecture from the checkpoint's embedded training config.

    The separately published YAML describes a different feature width.
    Only the released conditional model is supported; an unconditional
    checkpoint must not silently load into a different conditioning path.
    """
    model, diffusion = config["model"], config["diffusion"]
    if model["is_unconditional"]:
        raise ValueError("KumoAnomaly requires a conditional checkpoint")
    return {
        "num_features": model["target_dim"],
        "time_channels": model["timeemb"],
        "feature_channels": model["featureemb"],
        "channels": diffusion["channels"],
        "num_layers": diffusion["layers"],
        "num_heads": diffusion["nheads"],
        "embedding_channels": diffusion["diffusion_embedding_dim"],
        "num_steps": diffusion["num_steps"],
        "beta_start": diffusion["beta_start"],
        "beta_end": diffusion["beta_end"],
        "schedule": diffusion["schedule"],
    }


def remap_ckpt(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map upstream parameter names, preserving all keys for strict loading."""
    out = {}
    for key, value in state.items():
        if key.startswith("diffmodel."):
            key = "denoiser." + key.removeprefix("diffmodel.")
            key = key.replace("time_layer.layers.0.", "time_layer.")
            key = key.replace("feature_layer.layers.0.", "feature_layer.")
        elif key.startswith(
            ("embed_layer.", "embed_norm.", "time_embed_norm.")
        ):
            key = "side_embedding." + key
        out[key] = value
    return out
