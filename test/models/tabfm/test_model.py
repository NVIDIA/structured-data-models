from types import ModuleType
from typing import Any

import pytest
import torch
from sdm import CategoricalTensor, TableTensor
from sdm.models import TabFM
from sdm.models.tabfm.attention import MultiheadAttentionBlock
from sdm.models.tabfm.model import TabFMCore


def _make_models(
    upstream_tabfm_module: ModuleType,
    *,
    is_classifier: bool,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, TabFMCore]:
    config: dict[str, Any] = {
        "embed_dim": 8,
        "max_classes": 3,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 4,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 2,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
        "ff_factor": 2,
        "feature_group_size": 3,
        "num_freq": 4,
        "decoder_hidden": 16,
        "is_classifier": is_classifier,
    }
    upstream = upstream_tabfm_module.TabFM(**config)
    with torch.no_grad():
        upstream.cell_embedder.fourier_frequencies.normal_()
        upstream.cell_embedder.fourier_frequencies_cat.normal_()
    upstream = upstream.to(dtype).eval()
    model = TabFMCore(**config).to(dtype).eval()
    model.load_state_dict(upstream.state_dict(), strict=True)
    return upstream, model


def _set_chunk_sizes(
    model: torch.nn.Module,
    *,
    row: int | None,
    col: int | None,
    ffn: int | None,
) -> None:
    for module in model.modules():
        if hasattr(module, "row_chunk_size"):
            module.row_chunk_size = row  # ty: ignore[unresolved-attribute]
        if hasattr(module, "col_chunk_size"):
            module.col_chunk_size = col  # ty: ignore[unresolved-attribute]
        if hasattr(module, "ffn_chunk_size"):
            module.ffn_chunk_size = ffn  # ty: ignore[unresolved-attribute]


def _make_wrapper() -> TabFM:
    config: dict[str, Any] = {
        "embed_dim": 8,
        "max_classes": 4,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 4,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 2,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
        "ff_factor": 2,
        "feature_group_size": 3,
        "num_freq": 4,
        "decoder_hidden": 16,
    }
    return TabFM(
        cls_model=TabFMCore(**config, is_classifier=True),
        reg_model=TabFMCore(**config, is_classifier=False),
    )


@pytest.mark.parametrize("is_classifier", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_optional_routing", [False, True])
def test_tabfm_core_matches_upstream(
    upstream_tabfm_module: ModuleType,
    is_classifier: bool,
    dtype: torch.dtype,
    use_optional_routing: bool,
) -> None:
    upstream, model = _make_models(
        upstream_tabfm_module,
        is_classifier=is_classifier,
        dtype=dtype,
    )
    if use_optional_routing:
        _set_chunk_sizes(upstream, row=2, col=2, ffn=3)
        _set_chunk_sizes(model, row=2, col=2, ffn=3)
    else:
        _set_chunk_sizes(upstream, row=None, col=None, ffn=None)
        _set_chunk_sizes(model, row=None, col=None, ffn=None)

    batch_size, num_rows, num_features = 2, 6, 5
    input = torch.randn(batch_size, num_rows, num_features, dtype=dtype)
    input[0, 0, 0] = torch.nan
    if is_classifier:
        target = torch.randint(0, 3, (batch_size, num_rows))
        output_channels = 3
    else:
        target = torch.randn(batch_size, num_rows, dtype=dtype)
        output_channels = 1
    train_size = torch.tensor([4, 3], dtype=torch.long)
    cat_mask = None
    active_features = None
    if use_optional_routing:
        cat_mask = torch.tensor(
            [
                [False, True, False, True, False],
                [True, False, False, True, False],
            ]
        )
        active_features = torch.tensor([5, 3], dtype=torch.long)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)

    output = model(
        input,
        target,
        train_size,
        cat_mask=cat_mask,
        d=active_features,
    )
    expected = upstream(
        input,
        target,
        train_size,
        cat_mask=cat_mask,
        d=active_features,
    )

    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    assert output.shape == (batch_size, num_rows, output_channels)


def test_tabfm_core_uses_upstream_chunk_defaults() -> None:
    model = TabFMCore(
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        num_freq=4,
    )

    assert model.cell_embedder.row_chunk_size == 4096
    assert model.row_interactor.row_chunk_size == 4096
    assert model.row_interactor_2.row_chunk_size == 4096
    assert model.col_embedder.col_chunk_size == 16
    assert model.col_embedder_2.col_chunk_size == 16
    attention_blocks = [
        module
        for module in model.modules()
        if isinstance(module, MultiheadAttentionBlock)
    ]
    assert attention_blocks
    assert all(block.ffn_chunk_size == 8192 for block in attention_blocks)


@pytest.mark.parametrize("is_classifier", [False, True])
def test_tabfm_core_ignores_query_targets(is_classifier: bool) -> None:
    model = TabFMCore(
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        num_freq=4,
        is_classifier=is_classifier,
    ).eval()
    input = torch.randn(2, 6, 5)
    if is_classifier:
        target = torch.randint(0, 3, (2, 6))
        changed_query_target = (target + 1) % 3
    else:
        target = torch.randn(2, 6)
        changed_query_target = target + 100
    train_size = torch.tensor([4, 3], dtype=torch.long)
    row_index = torch.arange(input.size(1))[None, :]
    query_mask = row_index >= train_size[:, None]
    changed_query_target = torch.where(
        query_mask,
        changed_query_target,
        target,
    )

    baseline = model(input, target, train_size)
    output = model(input, changed_query_target, train_size)

    torch.testing.assert_close(output, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_tabfm_public_wrapper(
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
) -> None:
    model = _make_wrapper()
    num_rows, num_features, num_context = 7, 5, 4
    input = torch.randn(*batch_shape, num_rows, num_features)
    if dtype.is_floating_point:
        target = torch.randn(*batch_shape, num_context, dtype=dtype)
        output_channels = 1
    else:
        target = torch.randint(
            0,
            4,
            (*batch_shape, num_context),
            dtype=dtype,
        )
        output_channels = 4

    output = model(input, target)

    assert output.shape == (
        *batch_shape,
        num_rows - num_context,
        output_channels,
    )
    assert output.dtype == input.dtype
    if batch_shape:
        looped = torch.stack(
            [
                model(input[index], target[index])
                for index in range(input.size(0))
            ]
        )
        torch.testing.assert_close(output, looped)


def test_tabfm_public_wrapper_variable_train_size() -> None:
    model = _make_wrapper()
    input = torch.randn(2, 7, 5)
    target = torch.randint(0, 4, (2, 5))
    train_size = torch.tensor([5, 3])

    output = model(input, target, train_size=train_size)
    first = model(input[0], target[0], train_size=train_size[0])
    second = model(input[1], target[1], train_size=train_size[1])

    assert output.shape == (2, 4, 4)
    torch.testing.assert_close(output[0, :2], first)
    assert torch.count_nonzero(output[0, 2:]) == 0
    torch.testing.assert_close(output[1], second)


def test_tabfm_public_wrapper_preserves_table_categoricals() -> None:
    model = _make_wrapper()
    numerical = torch.randn(7, 2)
    categorical_data = torch.randint(0, 3, (7, 2), dtype=torch.int32)
    categorical = CategoricalTensor(
        data=categorical_data,
        categories=(torch.arange(3), torch.arange(3)),
    )
    table = TableTensor(
        columns={
            "numerical": ["a", "b"],
            "categorical": ["c", "d"],
        },
        numerical=numerical,
        categorical=categorical,
    )
    target = torch.randint(0, 4, (4,))
    tensor_input = torch.cat(
        [numerical, categorical_data.to(numerical.dtype)],
        dim=-1,
    )
    cat_mask = torch.tensor([False, False, True, True])

    output = model(table, target)
    expected = model(tensor_input, target, cat_mask=cat_mask)

    torch.testing.assert_close(output, expected)
    model.fit(table[:4], target)
    replayed = model.predict(table[4:])
    torch.testing.assert_close(replayed, output)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_tabfm_public_fit_predict_matches_forward(dtype: torch.dtype) -> None:
    model = _make_wrapper()
    input = torch.randn(2, 7, 5)
    if dtype.is_floating_point:
        target = torch.randn(2, 4, dtype=dtype)
    else:
        target = torch.randint(0, 4, (2, 4), dtype=dtype)
    cat_mask = torch.tensor(
        [
            [False, True, False, False, True],
            [False, True, False, False, True],
        ]
    )
    active_features = torch.tensor([5, 4])

    expected = model(
        input,
        target,
        cat_mask=cat_mask,
        d=active_features,
    )
    model.fit(
        input[:, :4],
        target,
        cat_mask=cat_mask,
        d=active_features,
    )
    output = model.predict(
        input[:, 4:],
        cat_mask=cat_mask,
        d=active_features,
    )

    torch.testing.assert_close(output, expected)
    model.clear()
    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(
            input[:, 4:],
            cat_mask=cat_mask,
            d=active_features,
        )


def test_tabfm_public_wrapper_construction_and_exports() -> None:
    from sdm.models.tabfm import TabFM as PackageTabFM

    assert PackageTabFM is TabFM
    assert repr(_make_wrapper()) == "TabFM()"
    with pytest.raises(ValueError, match="local checkpoint_path"):
        TabFM(pretrained=True)
    with pytest.raises(ValueError, match="requires pretrained=True"):
        TabFM(checkpoint_path="unused")
    with pytest.raises(ValueError, match="cls_model"):
        TabFM(cls_model=TabFMCore(is_classifier=False))
    with pytest.raises(ValueError, match="reg_model"):
        TabFM(reg_model=TabFMCore(is_classifier=True))
