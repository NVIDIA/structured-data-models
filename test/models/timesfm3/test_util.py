# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.timesfm3.util import (
    DecodeCache,
    get_activation_fn,
    get_output_patch_via_roll,
    get_running_stats,
    revin,
    stitch_patches,
    update_running_stats,
)
from sdm.testing import withCUDA


@withCUDA
def test_init_decode_cache(device: torch.device) -> None:
    caches = DecodeCache.init_decode_cache(
        num_layers=2,
        batch_size=2,
        num_variates=3,
        num_total_input_patches=5,
        num_heads=2,
        head_dim=4,
        device=device,
        dtype=torch.float64,
    )

    assert len(caches) == 2
    for cache in caches:
        assert cache.next_index.shape == (6,)
        assert cache.next_index.dtype == torch.int32
        assert cache.num_front_masked.shape == (6,)
        assert cache.num_front_masked.dtype == torch.int32
        assert cache.key.shape == (6, 5, 2, 4)
        assert cache.key.dtype == torch.float64
        assert cache.value.shape == cache.key.shape
        assert cache.value.dtype == cache.key.dtype
        assert cache.key.device == device
        assert cache.patch_mask is not None
        assert cache.patch_mask.shape == (6, 5)
        assert cache.patch_mask.all()
        assert cache.segment_ids is None
        assert cache.filled == 0
    assert caches[0].key.data_ptr() != caches[1].key.data_ptr()
    assert caches[0].value.data_ptr() != caches[1].value.data_ptr()


@withCUDA
def test_decode_cache_appends_heterogeneous_batch(
    device: torch.device,
) -> None:
    key_storage = torch.zeros(2, 6, 1, 2, device=device)
    value_storage = torch.zeros_like(key_storage)
    cache = DecodeCache(
        next_index=torch.tensor([1, 3], dtype=torch.int32, device=device),
        num_front_masked=torch.tensor(
            [1, 2],
            dtype=torch.int32,
            device=device,
        ),
        key=key_storage,
        value=value_storage,
    )
    key = torch.tensor(
        [
            [[[1.0, 2.0]], [[3.0, 4.0]]],
            [[[5.0, 6.0]], [[7.0, 8.0]]],
        ],
        device=device,
    )
    value = -key

    updated = cache.append(key, value)

    torch.testing.assert_close(
        updated.next_index,
        updated.next_index.new_tensor([3, 5]),
    )
    torch.testing.assert_close(
        updated.num_front_masked,
        updated.num_front_masked.new_tensor([1, 2]),
    )
    torch.testing.assert_close(updated.key[0, 1:3], key[0])
    torch.testing.assert_close(updated.key[1, 3:5], key[1])
    torch.testing.assert_close(updated.value[0, 1:3], value[0])
    torch.testing.assert_close(updated.value[1, 3:5], value[1])
    torch.testing.assert_close(updated.key[:, 5], torch.zeros_like(key[:, 0]))
    assert updated.key.data_ptr() == key_storage.data_ptr()
    assert updated.value.data_ptr() == value_storage.data_ptr()


@withCUDA
def test_decode_cache_appends_masks_and_segments(
    device: torch.device,
) -> None:
    cache = DecodeCache.init_decode_cache(
        num_layers=1,
        batch_size=2,
        num_variates=1,
        num_total_input_patches=4,
        num_heads=1,
        head_dim=2,
        device=device,
    )[0]
    key = torch.ones(2, 2, 1, 2, device=device)
    patch_mask = torch.tensor(
        [[True, False], [True, True]],
        device=device,
    )
    segment_ids = torch.tensor([[0, 0], [1, 1]], device=device)

    cache = cache.append(
        key,
        -key,
        patch_mask=patch_mask,
        segment_ids=segment_ids,
    )

    assert cache.patch_mask is not None
    assert cache.segment_ids is not None
    assert cache.filled == 2
    torch.testing.assert_close(
        cache.num_front_masked,
        cache.num_front_masked.new_tensor([1, 2]),
    )
    torch.testing.assert_close(cache.patch_mask[:, :2], patch_mask)
    torch.testing.assert_close(cache.segment_ids[:, :2], segment_ids)

    cache = cache.append(
        key[:, :1],
        -key[:, :1],
        patch_mask=torch.tensor([[False], [True]], device=device),
        segment_ids=torch.tensor([[0], [1]], device=device),
    )

    assert cache.filled == 3
    torch.testing.assert_close(
        cache.num_front_masked,
        cache.num_front_masked.new_tensor([1, 3]),
    )


@withCUDA
def test_update_running_stats(device: torch.device) -> None:
    x = torch.tensor(
        [[[1.0, 100.0, 2.0, 3.0]]],
        device=device,
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[[False, True, False, False]]],
        device=device,
    )
    initial = torch.zeros(1, 1, device=device, dtype=x.dtype)

    n, mu, sigma = update_running_stats(
        initial,
        initial,
        initial,
        x,
        mask,
    )

    assert n.dtype == x.dtype
    assert n.device == device
    torch.testing.assert_close(n, n.new_tensor([[3.0]]))
    torch.testing.assert_close(mu, mu.new_tensor([[2.0]]))
    torch.testing.assert_close(
        sigma,
        sigma.new_tensor([[2.0 / 3.0]]).sqrt(),
    )


@withCUDA
def test_update_running_stats_preserves_upstream_operation_order(
    device: torch.device,
) -> None:
    n = torch.tensor([[11.0]], device=device)
    mu = torch.tensor([[-92.03504943847656]], device=device)
    sigma = torch.tensor([[6.244617462158203]], device=device)
    x = torch.tensor(
        [[[-17.37119483947754, -76.65009307861328, 79.12385559082031]]],
        device=device,
    )
    mask = torch.zeros_like(x, dtype=torch.bool)

    _, actual_mu, actual_sigma = update_running_stats(n, mu, sigma, x, mask)

    inc_n = (~mask).float().sum(dim=-1)
    inc_mu = x.mean(dim=-1)
    inc_sigma = ((x - inc_mu.unsqueeze(-1)).square().mean(dim=-1)).sqrt()
    new_n = n + inc_n
    expected_sigma = (
        (
            n * sigma * sigma
            + inc_n * inc_sigma * inc_sigma
            + n * (mu - actual_mu) * (mu - actual_mu)
            + inc_n * (inc_mu - actual_mu) * (inc_mu - actual_mu)
        )
        / new_n
    ).sqrt()
    torch.testing.assert_close(
        actual_sigma,
        expected_sigma,
        rtol=0.0,
        atol=0.0,
    )


@withCUDA
def test_update_running_stats_fully_masked(device: torch.device) -> None:
    n = torch.tensor([[3.0]], device=device)
    mu = torch.tensor([[2.0]], device=device)
    sigma = torch.tensor([[1.0]], device=device)
    x = torch.tensor([[[100.0, 200.0]]], device=device)
    mask = torch.ones_like(x, dtype=torch.bool)

    actual = update_running_stats(n, mu, sigma, x, mask)

    for output, expected in zip(actual, (n, mu, sigma), strict=True):
        torch.testing.assert_close(output, expected)


@withCUDA
def test_get_running_stats_resets_segments(device: torch.device) -> None:
    values = torch.tensor(
        [[[[1.0, 3.0], [10.0, 10.0], [5.0, 7.0], [4.0, 6.0]]]],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 1, 1]], device=device)

    n, mu, sigma = get_running_stats(
        values,
        masks,
        segment_ids=segment_ids,
    )

    torch.testing.assert_close(n, n.new_tensor([[[2.0, 4.0, 2.0, 4.0]]]))
    torch.testing.assert_close(mu, mu.new_tensor([[[2.0, 6.0, 6.0, 5.5]]]))
    torch.testing.assert_close(
        sigma,
        sigma.new_tensor([[[1.0, 4.062019, 1.0, 1.118034]]]),
    )


@withCUDA
def test_get_running_stats_uses_initial_stats(device: torch.device) -> None:
    values = torch.tensor([[[[3.0, 5.0]]]], device=device)
    masks = torch.zeros_like(values, dtype=torch.bool)
    initial_stats = (
        values.new_tensor([[2.0]]),
        values.new_tensor([[1.0]]),
        values.new_tensor([[1.0]]),
    )

    n, mu, sigma = get_running_stats(
        values,
        masks,
        initial_stats=initial_stats,
    )

    torch.testing.assert_close(n, n.new_tensor([[[4.0]]]))
    torch.testing.assert_close(mu, mu.new_tensor([[[2.5]]]))
    torch.testing.assert_close(sigma, sigma.new_tensor([[[1.8027756]]]))


@withCUDA
def test_get_running_stats_uses_float32_for_bfloat16(
    device: torch.device,
) -> None:
    values = torch.ones(1, 1, 10, 1000, dtype=torch.bfloat16, device=device)
    masks = torch.zeros_like(values, dtype=torch.bool)

    n, mu, sigma = get_running_stats(values, masks)

    assert n.dtype == torch.float32
    assert mu.dtype == torch.float32
    assert sigma.dtype == torch.float32
    torch.testing.assert_close(n[..., -1], n.new_tensor([[10_000.0]]))
    torch.testing.assert_close(mu[..., -1], mu.new_tensor([[1.0]]))
    torch.testing.assert_close(sigma[..., -1], sigma.new_tensor([[0.0]]))


@withCUDA
def test_revin_round_trip(device: torch.device) -> None:
    x = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        device=device,
        dtype=torch.float64,
    )
    mu = torch.tensor([[2.0]], device=device, dtype=x.dtype)
    sigma = torch.tensor([[0.5]], device=device, dtype=x.dtype)

    normalized = revin(x, mu, sigma)
    restored = revin(normalized, mu, sigma, reverse=True)

    assert normalized.dtype == x.dtype
    assert normalized.device == device
    torch.testing.assert_close(restored, x)


@withCUDA
def test_revin_near_zero_sigma(device: torch.device) -> None:
    x = torch.tensor([[[2.0, 3.0]]], device=device)
    mu = torch.tensor([[2.0]], device=device)
    sigma = torch.tensor([[1e-7]], device=device)

    normalized = revin(x, mu, sigma)

    torch.testing.assert_close(
        normalized, normalized.new_tensor([[[0.0, 1.0]]])
    )


def test_revin_rejects_unsupported_shapes() -> None:
    x = torch.ones(2, 3, 4, 5)
    stats = torch.ones(2)

    with pytest.raises(ValueError, match="Unsupported shapes"):
        revin(x, stats, stats)


def test_revin_rejects_mismatched_stats() -> None:
    x = torch.ones(2, 3, 4)
    mu = torch.ones(2, 3)
    sigma = torch.ones(1, 3)

    with pytest.raises(
        ValueError, match="mu and sigma must have the same shape"
    ):
        revin(x, mu, sigma)


def test_revin_rejects_broadcastable_stats() -> None:
    x = torch.ones(2, 2, 3, 4)
    stats = torch.ones(1, 2, 3)

    with pytest.raises(ValueError, match="leading dimensions"):
        revin(x, stats, stats)


@withCUDA
def test_get_output_patch_via_roll(device: torch.device) -> None:
    x = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]],
        device=device,
    )

    output, wrap_mask = get_output_patch_via_roll(x, rolls=2)

    expected = x.new_tensor(
        [
            [
                [
                    [3.0, 4.0, 5.0, 6.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [7.0, 8.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0, 4.0],
                ]
            ]
        ]
    )
    expected_mask = torch.tensor(
        [
            [
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, True, True],
                    [True, True, True, True],
                ]
            ]
        ],
        device=device,
    )
    torch.testing.assert_close(output, expected)
    assert torch.equal(wrap_mask, expected_mask)


@pytest.mark.parametrize(
    ("num_patches", "rolls"),
    [(1, 2), (2, 3), (5, 6)],
)
@withCUDA
def test_get_output_patch_via_roll_varied_sizes(
    device: torch.device,
    num_patches: int,
    rolls: int,
) -> None:
    patch_len = 2
    x = torch.arange(
        num_patches * patch_len,
        device=device,
    ).reshape(1, 1, num_patches, patch_len)

    output, wrap_mask = get_output_patch_via_roll(x, rolls)

    expected = torch.stack(
        [
            torch.cat(
                [
                    x[:, :, (patch + roll) % num_patches, :]
                    for roll in range(1, rolls + 1)
                ],
                dim=-1,
            )
            for patch in range(num_patches)
        ],
        dim=2,
    )
    expected_mask = torch.tensor(
        [
            [patch + roll >= num_patches for roll in range(1, rolls + 1)]
            for patch in range(num_patches)
        ],
        device=device,
    ).repeat_interleave(patch_len, dim=-1)

    torch.testing.assert_close(output, expected)
    assert torch.equal(wrap_mask, expected_mask[None, None])


def test_get_activation_fn() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected_silu = x * x.sigmoid()

    torch.testing.assert_close(
        get_activation_fn("relu")(x),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(get_activation_fn("silu")(x), expected_silu)
    torch.testing.assert_close(get_activation_fn("swish")(x), expected_silu)
    assert get_activation_fn("none")(x) is x
    with pytest.raises(ValueError, match="swiglu"):
        get_activation_fn("swiglu")


@withCUDA
def test_stitch_patches(device: torch.device) -> None:
    patch_preds = torch.tensor(
        [
            [
                [
                    [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]],
                ]
            ]
        ],
        device=device,
        dtype=torch.float64,
    )

    output = stitch_patches(patch_preds, patch_len=3)

    expected = patch_preds.new_tensor(
        [[[[0.0], [1.0], [2.0], [3.0], [7.5], [12.0], [13.0], [14.0], [15.0]]]]
    )
    assert output.dtype == patch_preds.dtype
    assert output.device == device
    torch.testing.assert_close(output, expected)


@withCUDA
def test_stitch_single_patch(device: torch.device) -> None:
    patch_preds = torch.arange(6, device=device).reshape(1, 1, 1, 3, 2)

    output = stitch_patches(patch_preds, patch_len=2)

    assert torch.equal(output, patch_preds[:, :, 0])


@withCUDA
def test_decode_cache_appends_single_patch(device: torch.device) -> None:
    cache = DecodeCache(
        next_index=torch.tensor([1, 3], dtype=torch.int32, device=device),
        num_front_masked=torch.zeros(2, dtype=torch.int32, device=device),
        key=torch.zeros(2, 5, 1, 2, device=device),
        value=torch.zeros(2, 5, 1, 2, device=device),
    )
    key = torch.tensor(
        [[[[1.0, 2.0]]], [[[3.0, 4.0]]]],
        device=device,
    )

    updated = cache.append(key, -key)

    torch.testing.assert_close(
        updated.next_index,
        updated.next_index.new_tensor([2, 4]),
    )
    torch.testing.assert_close(updated.key[0, 1], key[0, 0])
    torch.testing.assert_close(updated.key[1, 3], key[1, 0])
    torch.testing.assert_close(updated.value[0, 1], -key[0, 0])
    torch.testing.assert_close(updated.value[1, 3], -key[1, 0])
