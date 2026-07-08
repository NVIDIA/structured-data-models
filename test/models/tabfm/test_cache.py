import pytest
import torch
from sdm.cache import Cache, KVCacheEntry
from sdm.models import TabFM
from sdm.models.tabfm.attention import Encoder, MultiheadAttention
from sdm.models.tabfm.model import TabFMCore


def _core(*, is_classifier: bool, dtype: torch.dtype) -> TabFMCore:
    return TabFMCore(
        embed_dim=8,
        max_classes=4,
        col_num_blocks=2,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        num_freq=4,
        decoder_hidden=16,
        is_classifier=is_classifier,
        dtype=dtype,
    )


def _wrapper(dtype: torch.dtype) -> TabFM:
    return TabFM(
        cls_model=_core(is_classifier=True, dtype=dtype),
        reg_model=_core(is_classifier=False, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tabfm_attention_projected_kv_cache(dtype: torch.dtype) -> None:
    attention = MultiheadAttention(
        channels=16,
        num_heads=4,
        dtype=dtype,
    ).eval()
    query = torch.randn(2, 3, 16, dtype=dtype)
    context = torch.randn(2, 5, 16, dtype=dtype)

    expected, key_value = attention(
        query,
        context,
        context,
        return_key_value=True,
    )
    output = attention(query, key_value)

    assert isinstance(key_value, KVCacheEntry)
    assert key_value.key.shape == (2, 5, 4, 4)
    assert key_value.value.shape == (2, 5, 4, 4)
    torch.testing.assert_close(output, expected)


def test_tabfm_encoder_cache_rejects_rope() -> None:
    encoder = Encoder(
        num_blocks=1,
        channels=16,
        num_heads=4,
        feedforward_channels=32,
    )

    with pytest.raises(ValueError, match="RoPE"):
        encoder(torch.randn(2, 5, 16), cache=Cache())


@pytest.mark.parametrize("is_classifier", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_columns", [False, True])
def test_tabfm_fit_predict_uses_projected_context_cache(
    is_classifier: bool,
    dtype: torch.dtype,
    chunk_columns: bool,
) -> None:
    model = _wrapper(dtype)
    if chunk_columns:
        for core in (model.cls_model, model.reg_model):
            core.col_embedder.col_chunk_size = 3
            core.col_embedder_2.col_chunk_size = 3
    context = torch.randn(2, 4, 5, dtype=dtype)
    first_query = torch.randn(2, 3, 5, dtype=dtype)
    second_query = torch.randn(2, 2, 5, dtype=dtype)
    if is_classifier:
        target = torch.randint(0, 4, (2, 4))
    else:
        target = torch.randn(2, 4, dtype=dtype)
    cat_mask = torch.tensor([False, True, False, False, True])
    active_features = torch.tensor([5, 4])

    expected_first = model(
        torch.cat([context, first_query], dim=1),
        target,
        cat_mask=cat_mask,
        d=active_features,
    )
    expected_second = model(
        torch.cat([context, second_query], dim=1),
        target,
        cat_mask=cat_mask,
        d=active_features,
    )
    model.fit(
        context,
        target,
        cat_mask=cat_mask,
        d=active_features,
    )

    assert model._cache is not None
    assert "x" not in model._cache
    assert "y" not in model._cache
    key_values = [
        value
        for value in model._cache.values()
        if isinstance(value, KVCacheEntry)
    ]
    assert len(key_values) == 6
    with pytest.raises(RuntimeError, match="record"):
        model._cache["new"] = torch.tensor(1)

    first_output = model.predict(
        first_query,
        cat_mask=cat_mask,
        d=active_features,
    )
    second_output = model.predict(
        second_query,
        cat_mask=cat_mask,
        d=active_features,
    )

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(
        first_output,
        expected_first,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        second_output,
        expected_second,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("is_classifier", [False, True])
def test_tabfm_core_cache_supports_mixed_context_lengths(
    is_classifier: bool,
) -> None:
    model = _core(is_classifier=is_classifier, dtype=torch.float32).eval()
    input = torch.randn(2, 7, 5)
    if is_classifier:
        target = torch.randint(0, 4, (2, 7))
    else:
        target = torch.randn(2, 7)
    train_size = torch.tensor([4, 3])
    expected = model(input, target, train_size)

    context = input[:, :4].clone()
    context[1, 3] = 0
    context_target = target[:, :4].clone()
    context_target[1, 3] = -100
    cache = Cache()
    model(
        context,
        context_target,
        train_size,
        cache=cache,
    )
    cache.freeze()

    query = input.new_zeros(2, 4, 5)
    query[0, :3] = input[0, 4:]
    query[1] = input[1, 3:]
    query_target = target.new_full((2, 4), -100)
    output = model(
        query,
        query_target,
        torch.zeros(2, dtype=torch.long),
        cache=cache,
    )

    torch.testing.assert_close(output[0, :3], expected[0, 4:])
    torch.testing.assert_close(output[1], expected[1, 3:])


def test_tabfm_cache_rejects_changed_metadata() -> None:
    model = _wrapper(torch.float32)
    context = torch.randn(2, 4, 5)
    target = torch.randint(0, 4, (2, 4))
    cat_mask = torch.tensor([False, True, False, False, True])
    active_features = torch.tensor([5, 4])
    model.fit(
        context,
        target,
        cat_mask=cat_mask,
        d=active_features,
    )
    query = torch.randn(2, 2, 5)

    with pytest.raises(ValueError, match="cat_mask"):
        model.predict(query, cat_mask=~cat_mask, d=active_features)
    with pytest.raises(ValueError, match="query d"):
        model.predict(cat_mask=cat_mask, d=torch.tensor([5, 5]), x=query)

    model.to(torch.bfloat16)
    with pytest.raises(ValueError, match="dtype changed"):
        model.predict(
            query.to(torch.bfloat16),
            cat_mask=cat_mask,
            d=active_features,
        )
