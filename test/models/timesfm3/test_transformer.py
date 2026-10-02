# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.timesfm3.transformer import make_attn_mask
from sdm.testing import withCUDA


@withCUDA
def test_make_attn_mask(device: torch.device) -> None:
    patch_mask = torch.tensor(
        [[True, False, False], [False, True, False]], device=device
    )
    mask = make_attn_mask(patch_mask)
    causal = torch.ones(3, 3, dtype=torch.bool, device=device).tril()
    expected = causal[None, None] & ~patch_mask[:, None, None, :]
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_make_attn_mask_noncausal(device: torch.device) -> None:
    patch_mask = torch.tensor([[True, False, False]], device=device)
    mask = make_attn_mask(patch_mask, causal=False)
    torch.testing.assert_close(mask, ~patch_mask[:, None, None, :])
