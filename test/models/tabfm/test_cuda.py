from collections.abc import Callable
from types import ModuleType

import pytest
import torch
from sdm.cache import Cache
from sdm.models import TabFM
from sdm.models.tabfm.model import TabFMCore
from sdm.testing import onlyCUDA


def _core(
    *,
    is_classifier: bool,
    dtype: torch.dtype,
    chunk_columns: bool,
) -> TabFMCore:
    model = TabFMCore(
        embed_dim=8,
        max_classes=4,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        num_freq=4,
        decoder_hidden=16,
        is_classifier=is_classifier,
        device="cuda",
        dtype=dtype,
    ).eval()
    model.col_embedder.col_chunk_size = 3 if chunk_columns else None
    model.col_embedder_2.col_chunk_size = 3 if chunk_columns else None
    return model


def _skip_unsupported_dtype(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-6
    return 2e-2, 2e-2


@onlyCUDA
@pytest.mark.parametrize("is_classifier", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
)
@pytest.mark.parametrize("chunk_columns", [False, True])
def test_tabfm_cuda_upstream_and_cache_parity(
    upstream_tabfm_module: ModuleType,
    is_classifier: bool,
    dtype: torch.dtype,
    chunk_columns: bool,
) -> None:
    _skip_unsupported_dtype(dtype)
    model = _core(
        is_classifier=is_classifier,
        dtype=dtype,
        chunk_columns=chunk_columns,
    )
    upstream = upstream_tabfm_module.TabFM(
        embed_dim=8,
        max_classes=4,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        num_freq=4,
        decoder_hidden=16,
        is_classifier=is_classifier,
    ).to(device="cuda", dtype=dtype)
    upstream.load_state_dict(model.state_dict(), strict=True)
    upstream.eval()
    upstream.col_embedder.col_chunk_size = 3 if chunk_columns else None
    upstream.col_embedder_2.col_chunk_size = 3 if chunk_columns else None

    input = torch.randn(2, 7, 5, device="cuda", dtype=dtype)
    if is_classifier:
        target = torch.randint(0, 4, (2, 7), device="cuda")
    else:
        target = torch.randn(2, 7, device="cuda", dtype=dtype)
    train_size = torch.tensor([4, 3], device="cuda")
    cat_mask = torch.tensor(
        [False, True, False, False, True],
        device="cuda",
    )
    active_features = torch.tensor([5, 4], device="cuda")

    expected = upstream(
        input,
        target,
        train_size,
        cat_mask=cat_mask,
        d=active_features,
    )
    output = model(
        input,
        target,
        train_size,
        cat_mask=cat_mask,
        d=active_features,
    )
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    context = input[:, :4].clone()
    context[1, 3] = 0
    context_target = target[:, :4].clone()
    context_target[1, 3] = -100
    cache = Cache()
    model(
        context,
        context_target,
        train_size,
        cat_mask=cat_mask,
        d=active_features,
        cache=cache,
    )
    cache.freeze()
    query = input.new_zeros(2, 4, 5)
    query[0, :3] = input[0, 4:]
    query[1] = input[1, 3:]
    query_target = target.new_full((2, 4), -100)
    cached = model(
        query,
        query_target,
        torch.zeros(2, dtype=torch.long, device="cuda"),
        cat_mask=cat_mask,
        d=active_features,
        cache=cache,
    )

    torch.testing.assert_close(
        cached[0, :3],
        output[0, 4:],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        cached[1],
        output[1, 3:],
        rtol=rtol,
        atol=atol,
    )


def _measure_peak_bytes(function: Callable[[], object]) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    function()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


@onlyCUDA
@pytest.mark.parametrize("chunk_columns", [False, True])
def test_tabfm_cuda_records_peak_memory(
    record_property: Callable[[str, object], None],
    chunk_columns: bool,
) -> None:
    cls_model = _core(
        is_classifier=True,
        dtype=torch.float32,
        chunk_columns=chunk_columns,
    )
    reg_model = _core(
        is_classifier=False,
        dtype=torch.float32,
        chunk_columns=chunk_columns,
    )
    model = TabFM(cls_model=cls_model, reg_model=reg_model)
    context = torch.randn(2, 8, 8, device="cuda")
    target = torch.randint(0, 4, (2, 8), device="cuda")
    query = torch.randn(2, 4, 8, device="cuda")
    second_query = torch.randn(2, 2, 8, device="cuda")
    combined = torch.cat([context, query], dim=1)

    uncached_bytes = _measure_peak_bytes(lambda: model(combined, target))
    prefill_bytes = _measure_peak_bytes(lambda: model.fit(context, target))
    replay_bytes = _measure_peak_bytes(lambda: model.predict(query))
    second_expected = model(
        torch.cat([context, second_query], dim=1),
        target,
    )
    second_cached = model.predict(second_query)

    torch.testing.assert_close(second_cached, second_expected)

    record_property("device", torch.cuda.get_device_name())
    record_property("torch_version", torch.__version__)
    record_property("cuda_version", torch.version.cuda or "unknown")
    record_property("dtype", "float32")
    record_property("table_shape", tuple(combined.shape))
    record_property("context_rows", context.size(1))
    record_property("query_rows", query.size(1))
    record_property("column_chunk_size", 3 if chunk_columns else None)
    record_property("uncached_peak_bytes", uncached_bytes)
    record_property("prefill_peak_bytes", prefill_bytes)
    record_property("replay_peak_bytes", replay_bytes)

    assert uncached_bytes > 0
    assert prefill_bytes > 0
    assert replay_bytes > 0
