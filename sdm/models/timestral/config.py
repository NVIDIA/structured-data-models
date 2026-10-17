# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

LAYER_KIND_AXES: dict[str, tuple[bool, ...]] = {
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
    "variate_time": (False, True),
}


@dataclass(frozen=True)
class _TimestralConfig:
    patch_size: int
    d_model: int
    num_heads: int
    mlp_hidden_dim: int
    block_schedule: tuple[str, ...]
    quantile_knots: tuple[float, ...]
    block_size: int
    context_length: int
    rope_theta: float = 10000.0
    rope_partial_factor: float = 0.5
    norm_eps: float = 1e-06
    attn_bias: bool = False
    mlp_bias: bool = False
    time_attn_gate: str = "sdpa_sigmoid"
    variate_attn_gate: str = "sdpa_sigmoid"
    minimum_scale: float = 1e-06
    use_asinh: bool = True
    stabilize_with_global: bool = False
    covariate_enabled: bool = True
    split_context_horizon_attn: bool = True
    attend_missing_patches: bool = True
    quantile_width_factor: float = 2.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.rope_partial_factor)

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_knots)

    @property
    def median_index(self) -> int:
        return self.quantile_knots.index(0.5)
