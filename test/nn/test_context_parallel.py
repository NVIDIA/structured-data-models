import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.nn import SDPA, context_parallel_attention
from sdm.nn.context_parallel import (
    combine,
    combine_partials,
    combine_scope,
    context_group,
    context_parallel,
    partial_attention,
)
from sdm.testing import onlyCUDA


def full_attention(
    query: Tensor,  # [..., Q, H, C]
    key: Tensor,  # [..., KV, Hkv, C]
    value: Tensor,  # [..., KV, Hkv, C]
    scale: float | None = None,
) -> Tensor:  # [..., Q, H, C]
    groups = query.size(-2) // key.size(-2)
    key = key.repeat_interleave(groups, dim=-2)
    value = value.repeat_interleave(groups, dim=-2)
    return F.scaled_dot_product_attention(
        query=query.transpose(-3, -2),
        key=key.transpose(-3, -2),
        value=value.transpose(-3, -2),
        scale=scale,
    ).transpose(-3, -2)


@onlyCUDA
def test_partial_attention_returns_scores_log_sum_exp():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    query = torch.randn(2, 16, 4, 32, device=device)
    key = torch.randn(2, 64, 4, 32, device=device)
    value = torch.randn(2, 64, 4, 32, device=device)

    out, lse = partial_attention(query, key, value)

    scale = query.size(-1) ** -0.5
    scores = torch.einsum("bqhc,bkhc->bhqk", query, key) * scale
    torch.testing.assert_close(
        lse, scores.logsumexp(dim=-1).transpose(-1, -2)
    )
    torch.testing.assert_close(out, full_attention(query, key, value))


@onlyCUDA
@pytest.mark.parametrize("num_shards", [1, 2, 4, 8, 7, 13])
def test_combine_partials_matches_full_attention(num_shards):
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    query = torch.randn(2, 128, 8, 64, device=device)
    key = torch.randn(2, 3000, 8, 64, device=device)
    value = torch.randn(2, 3000, 8, 64, device=device)

    partials = [
        partial_attention(query, key_shard, value_shard)
        for key_shard, value_shard in zip(
            key.chunk(num_shards, dim=-3), value.chunk(num_shards, dim=-3)
        )
    ]
    out = combine_partials(partials)

    torch.testing.assert_close(
        out, full_attention(query, key, value), rtol=1e-4, atol=1e-4
    )


@onlyCUDA
def test_context_parallel_attention_grouped_query_and_scale():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    query = torch.randn(2, 64, 8, 64, device=device)  # 8 query heads
    key = torch.randn(2, 2000, 2, 64, device=device)  # 2 kv heads (GQA)
    value = torch.randn(2, 2000, 2, 64, device=device)
    scale = 0.05

    ref = full_attention(query, key, value, scale=scale)

    # No group -> attends to the full local context (identity reduction).
    out = context_parallel_attention(
        query, key, value, group=None, scale=scale
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    # Sharded key/value recombine exactly under grouped-query attention.
    sharded = combine_partials(
        [
            partial_attention(query, key_shard, value_shard, scale=scale)
            for key_shard, value_shard in zip(
                key.chunk(4, dim=-3), value.chunk(4, dim=-3)
            )
        ]
    )
    torch.testing.assert_close(sharded, ref, rtol=1e-4, atol=1e-4)


def test_combine_without_group_is_identity():
    out = torch.randn(2, 8, 4, 16)
    lse = torch.randn(2, 8, 4)
    torch.testing.assert_close(combine(out, lse, group=None), out)


def test_context_managers_default_to_no_group():
    assert context_group() is None
    with context_parallel(None):
        assert context_group() is None


@onlyCUDA
def test_combine_scope_none_leaves_sdpa_unchanged():
    # A None combine group must be inert: SDPA behaves exactly as usual.
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    sdpa = SDPA(num_query_heads=8).to(device)
    query = torch.randn(2, 64, 8, 32, device=device)
    key = torch.randn(2, 128, 8, 32, device=device)
    value = torch.randn(2, 128, 8, 32, device=device)

    reference = sdpa(query, key, value)
    with combine_scope(None):
        out = sdpa(query, key, value)
    torch.testing.assert_close(out, reference)
