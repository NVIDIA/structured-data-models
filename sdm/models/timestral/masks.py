# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def _compute_enc_key_validity(
    obs_per_patch: Tensor, attend_missing: bool
) -> Tensor:
    if not attend_missing:
        return obs_per_patch
    started = obs_per_patch.cummax(dim=-1).values
    return obs_per_patch | started


def _enc_causal_mask(key_valid: Tensor) -> Tensor:
    _, ll = key_valid.shape
    device = key_valid.device
    causal = torch.tril(torch.ones(ll, ll, dtype=torch.bool, device=device))
    eye = torch.eye(ll, dtype=torch.bool, device=device)
    return causal[None] & key_valid[:, None, :] | eye[None]


def _build_forecast_time_mask(
    context_len: int, horizon: int, key_valid: Tensor, device: torch.device
) -> Tensor:
    rows = key_valid.shape[0]
    enc = _enc_causal_mask(key_valid)
    enc_dec = torch.zeros(
        rows, context_len, horizon, dtype=torch.bool, device=device
    )
    dec_enc = key_valid[:, None, :].expand(rows, horizon, context_len)
    dec_dec = torch.tril(
        torch.ones(horizon, horizon, dtype=torch.bool, device=device)
    )[None].expand(rows, horizon, horizon)
    top = torch.cat([enc, enc_dec], dim=2)
    bottom = torch.cat([dec_enc, dec_dec], dim=2)
    return torch.cat([top, bottom], dim=1).unsqueeze(1)


def _build_forecast_positions(
    context_len: int, horizon: int, first_obs: Tensor
) -> Tensor:
    p = torch.arange(context_len + horizon, device=first_obs.device)
    return p[None, :] - first_obs[:, None]


def _compute_block_layout(
    min_context: Tensor, n_patches: int, block_size: int
) -> tuple[Tensor, Tensor]:
    device = min_context.device
    p = torch.arange(n_patches, device=device)[None, :]
    s = min_context[:, None]
    active = p >= s
    offset = (p - s).clamp(min=0)
    anchor = s + block_size * (offset // block_size)
    anchor = torch.where(active, anchor, torch.zeros_like(anchor))
    return (anchor, active)


def _build_two_stream_time_mask(
    min_context: Tensor,
    n_patches: int,
    block_size: int,
    key_valid: Tensor,
    bidirectional_rows: Tensor | None,
) -> Tensor:
    anchor, active = _compute_block_layout(min_context, n_patches, block_size)
    rows, n = active.shape
    device = active.device
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    key_col = key_valid[:, None, :]
    j = torch.arange(n, device=device).view(1, 1, n)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    enc_enc = _enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        enc_bi = key_valid[:, None, :].expand(rows, n, n) | eye[None]
        enc_enc = torch.where(
            bidirectional_rows[:, None, None], enc_bi, enc_enc
        )
    enc_dec = torch.zeros(rows, n, n, dtype=torch.bool, device=device)
    dec_enc = active[:, :, None] & (j < anchor[:, :, None]) & key_col
    same_block = anchor[:, :, None] == anchor[:, None, :]
    dec_dec = (
        same_block & causal[None] & active[:, :, None] & active[:, None, :]
    )
    dec_dec = dec_dec | eye[None]
    top = torch.cat([enc_enc, enc_dec], dim=2)
    bottom = torch.cat([dec_enc, dec_dec], dim=2)
    return torch.cat([top, bottom], dim=1).unsqueeze(1)


def _variate_firewall(
    r: int,
    device: torch.device,
    target_rows: Tensor | None,
    bidirectional_rows: Tensor | None,
) -> Tensor | None:
    if target_rows is None and bidirectional_rows is None:
        return None
    forbid = torch.zeros(r, r, dtype=torch.bool, device=device)
    if target_rows is not None:
        forbid = forbid | ~target_rows[:, None] & target_rows[None, :]
    if bidirectional_rows is not None:
        forbid = (
            forbid
            | bidirectional_rows[:, None] & (~bidirectional_rows)[None, :]
        )
    return forbid


def _build_two_stream_variate_mask(
    group_ids: Tensor,
    enc_key_valid: Tensor,
    n_dec: int,
    target_rows: Tensor | None = None,
    bidirectional_rows: Tensor | None = None,
) -> Tensor | None:
    r, _ = enc_key_valid.shape
    if r == 1:
        return None
    device = enc_key_valid.device
    row_eq = group_ids[:, None] == group_ids[None, :]
    dec_valid = torch.ones(r, n_dec, dtype=torch.bool, device=device)
    slot_valid = torch.cat([enc_key_valid, dec_valid], dim=1)
    key_valid = slot_valid.permute(1, 0)
    mask = row_eq[None, None] & key_valid[:, None, None, :]
    forbid = _variate_firewall(r, device, target_rows, bidirectional_rows)
    if forbid is not None:
        mask = mask & ~forbid[None, None]
    eye = torch.eye(r, dtype=torch.bool, device=device)
    return mask | eye[None, None]


def _build_two_stream_positions(n_patches: int, first_obs: Tensor) -> Tensor:
    p = torch.arange(n_patches, device=first_obs.device)
    row = p[None, :] - first_obs[:, None]
    return torch.cat([row, row], dim=-1)
