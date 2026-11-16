# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parameter mapping for the released missingness-aware checkpoint."""

from torch import Tensor

from sdm.models.kumo.timeseries.forecast.ckpt import (
    remap_ckpt as remap_forecast,
)


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map the released T5, masking, normalization, and channel parameters."""
    out = {
        key.replace("cross_channel.block.attn.", "cross_channel.attn."): value
        for key, value in remap_forecast(ckpt).items()
    }
    out.update(
        mask_token=ckpt["mask_token"],
        mask_embedding=ckpt["patch_embedding.mask_embedding"],
    )
    out["normalizer.prior"] = ckpt["normalizer._global_stdev"].squeeze(0)
    out["normalizer.initialized"] = ckpt[
        "normalizer._global_stdev_initialized"
    ]
    for key, value in ckpt.items():
        if key.startswith(("obs_attn_bias.", "missingness_adapter.")):
            out[key] = value
    for suffix in ("residual_gate", "channel_embed.weight"):
        out["cross_channel." + suffix] = ckpt["cross_channel_attn." + suffix]
    # The checkpoint's sinusoidal position table is deterministic; SDM builds
    # the same positions on the input device instead of storing 5,000 rows.
    return out
