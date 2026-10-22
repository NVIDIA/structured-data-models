# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.nn import CausalPatchNorm
from sdm.testing import withCUDA


@withCUDA
def test_causal_patch_norm(device: torch.device) -> None:
    x = torch.tensor([[1.0, 3.0, float("nan"), 7.0, 9.0, 11.0]], device=device)
    observed = x.isfinite()
    norm = CausalPatchNorm(patch_size=2, use_asinh=True)
    out, mean, scale = norm(x, observed)
    for end in (2, 4, 6):
        values = x[0, :end][observed[0, :end]]
        torch.testing.assert_close(
            mean[0, end - 2 : end], values.mean().expand(2)
        )
        torch.testing.assert_close(
            scale[0, end - 2 : end], values.std().expand(2)
        )
    assert out[0, 2] == 0
    torch.testing.assert_close(
        out.sinh() * scale + mean, x.masked_fill(~observed, mean[0, 2])
    )
    changed = x.clone()
    changed[..., 4:] += 100
    torch.testing.assert_close(
        norm(changed, observed)[0][..., :4], out[..., :4]
    )
