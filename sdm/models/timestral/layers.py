# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sdm.models.timestral.config import LAYER_KIND_AXES
from sdm.nn import SDPA, RMSNorm, RotaryEmbedding, SwiGLU


@dataclass(frozen=True)
class _TwoStreamMask:
    mask: Tensor
    enc_len: int


class _Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_time: bool,
        use_bias: bool,
        gate: str,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_time = is_time
        self.sdpa = SDPA(num_query_heads=num_heads)
        self.qkv_lin = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_lin = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.gate_lin = (
            nn.Linear(embed_dim, embed_dim, bias=use_bias)
            if gate == "sdpa_sigmoid"
            else None
        )

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        qkv = self.qkv_lin(x).unflatten(-1, (3, self.num_heads, self.head_dim))
        return qkv.unbind(-3)

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        return self.sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            is_causal=is_causal,
        ).flatten(-2)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | _TwoStreamMask | None,
        positions: Tensor,
        rope: RotaryEmbedding,
    ) -> Tensor:
        rows, num_patches, _ = x.shape
        inputs = x if self.is_time else x.permute(1, 0, 2)
        q, k, v = self._qkv(inputs)
        if self.is_time:
            # Queries and keys share positions and rotary frequencies.
            q, k = rope(
                torch.stack([q, k]),
                position_ids=positions,
            ).unbind(0)
        if isinstance(mask, _TwoStreamMask):
            n = mask.enc_len
            out_enc = self._attend(
                query=q[:, :n],
                key=k[:, :n],
                value=v[:, :n],
                mask=mask.mask[:, :, :n, :n],
            )
            out_dec = self._attend(q[:, n:], k, v, mask.mask[:, :, n:, :])
            out = torch.cat([out_enc, out_dec], dim=1)
        else:
            out = self._attend(
                query=q,
                key=k,
                value=v,
                mask=mask,
                is_causal=self.is_time and mask is None,
            )
        if self.gate_lin is not None:
            out = out * self.gate_lin(inputs).sigmoid()
        if self.is_time:
            out = out.reshape(rows, num_patches, self.embed_dim)
        else:
            out = out.reshape(num_patches, rows, self.embed_dim).permute(
                1, 0, 2
            )
        return self.out_lin(out)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        kind: str,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        axes = LAYER_KIND_AXES[kind]
        self.attn_norms = nn.ModuleList(
            [RMSNorm(embed_dim, eps=norm_eps) for _ in axes]
        )
        self.attns = nn.ModuleList(
            [
                _Attention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    is_time=is_time,
                    use_bias=attn_bias,
                    gate=time_attn_gate if is_time else variate_attn_gate,
                )
                for is_time in axes
            ]
        )
        self.mlp_norm = RMSNorm(embed_dim, eps=norm_eps)
        self.mlp = SwiGLU(embed_dim, mlp_hidden_dim, bias=mlp_bias)

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor | _TwoStreamMask | None,
        variate_mask: Tensor | None,
        positions: Tensor,
        rope: RotaryEmbedding,
    ) -> Tensor:
        for norm, attn in zip(self.attn_norms, self.attns):
            x = x + attn(
                x=norm(x),
                mask=time_mask if attn.is_time else variate_mask,
                positions=positions,
                rope=rope,
            )
        return x + self.mlp(self.mlp_norm(x))


class _Transformer(nn.Module):
    def __init__(
        self,
        block_schedule: tuple[str, ...],
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rope_theta: float,
        rope_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    kind=kind,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    norm_eps=norm_eps,
                    attn_bias=attn_bias,
                    mlp_bias=mlp_bias,
                    time_attn_gate=time_attn_gate,
                    variate_attn_gate=variate_attn_gate,
                )
                for kind in block_schedule
            ]
        )
        self.rope = RotaryEmbedding(
            channels=embed_dim // num_heads,
            layout="split_half",
            theta=rope_theta,
            partial_rotary_factor=rope_dim / (embed_dim // num_heads),
            requires_grad=False,
        )
        self.out_norm = RMSNorm(embed_dim, eps=norm_eps)

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor | _TwoStreamMask | None,
        variate_mask: Tensor | None,
        positions: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                time_mask=time_mask,
                variate_mask=variate_mask,
                positions=positions,
                rope=self.rope,
            )
        return self.out_norm(x)
