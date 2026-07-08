from types import ModuleType

import pytest
import torch
from sdm.models.tabfm.embedding import (
    CellEmbedder,
    ColEmbedding,
    InducedSelfAttentionBlock,
    SetTransformer,
)
from sdm.models.tabfm.mlp import MLP


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mlp_matches_upstream(
    upstream_tabfm_module: ModuleType,
    activation: str,
    dtype: torch.dtype,
) -> None:
    upstream = upstream_tabfm_module.MLP(
        in_dim=4,
        hidden_dims=[6, 8],
        out_dim=3,
        activation=activation,
    ).to(dtype)
    model = MLP(
        in_channels=4,
        hidden_channels=[6, 8],
        out_channels=3,
        activation=activation,
    ).to(dtype)
    model.load_state_dict(upstream.state_dict())
    input = torch.randn(2, 5, 4, dtype=dtype)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)

    torch.testing.assert_close(
        model(input),
        upstream(input),
        rtol=rtol,
        atol=atol,
    )


def test_cell_embedder_group_offsets_and_active_width() -> None:
    model = CellEmbedder(
        channels=8,
        max_classes=3,
        feature_group_size=3,
        num_frequencies=4,
    )
    input = torch.arange(5, dtype=torch.float32).view(1, 1, 5)

    grouped = model._group(input)
    expected = torch.tensor(
        [
            [
                [
                    [0, 1, 3],
                    [1, 2, 4],
                    [2, 3, 0],
                    [3, 4, 1],
                    [4, 0, 2],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(grouped, expected)

    active_grouped = model._group(input, d=torch.tensor([3]))
    active_expected = torch.tensor(
        [
            [
                [
                    [0, 1, 0],
                    [1, 2, 1],
                    [2, 0, 2],
                    [0, 1, 0],
                    [1, 2, 1],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(active_grouped, active_expected)


@pytest.mark.parametrize("is_classifier", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_cat_mask", [False, True])
@pytest.mark.parametrize("use_active_features", [False, True])
@pytest.mark.parametrize("row_chunk_size", [None, 2])
def test_cell_embedder_matches_upstream(
    upstream_tabfm_module: ModuleType,
    is_classifier: bool,
    dtype: torch.dtype,
    use_cat_mask: bool,
    use_active_features: bool,
    row_chunk_size: int | None,
) -> None:
    upstream = upstream_tabfm_module.CellEmbedder(
        embed_dim=8,
        max_classes=3,
        feature_group_size=3,
        num_freq=4,
        is_classifier=is_classifier,
    )
    with torch.no_grad():
        upstream.fourier_frequencies.normal_()
        upstream.fourier_frequencies_cat.normal_()
    upstream = upstream.to(dtype)

    model = CellEmbedder(
        channels=8,
        max_classes=3,
        feature_group_size=3,
        num_frequencies=4,
        is_classifier=is_classifier,
    ).to(dtype)
    model.load_state_dict(upstream.state_dict())
    upstream.row_chunk_size = row_chunk_size
    model.row_chunk_size = row_chunk_size

    batch_size, num_rows, num_features = 2, 6, 5
    input = torch.randn(
        batch_size,
        num_rows,
        num_features,
        dtype=dtype,
    )
    if is_classifier:
        target = torch.randint(0, 3, (batch_size, num_rows))
    else:
        target = torch.randn(batch_size, num_rows, dtype=dtype)
    train_size = torch.tensor([4, 3], dtype=torch.long)
    cat_mask = None
    if use_cat_mask:
        cat_mask = torch.tensor(
            [
                [False, True, False, True, False],
                [True, False, False, True, False],
            ]
        )
    active_features = None
    if use_active_features:
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
    if use_active_features:
        assert torch.count_nonzero(output[1, :, 3:]) == 0


@pytest.mark.parametrize("is_classifier", [False, True])
def test_cell_embedder_ignores_query_targets(
    is_classifier: bool,
) -> None:
    model = CellEmbedder(
        channels=8,
        max_classes=3,
        feature_group_size=3,
        num_frequencies=4,
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


def test_cell_embedder_context_targets_affect_context_rows() -> None:
    model = CellEmbedder(
        channels=8,
        max_classes=3,
        feature_group_size=3,
        num_frequencies=4,
    ).eval()
    input = torch.randn(2, 6, 5)
    target = torch.randint(0, 3, (2, 6))
    changed_target = target.clone()
    changed_target[0, 0] = (changed_target[0, 0] + 1) % 3
    train_size = torch.tensor([4, 3], dtype=torch.long)

    baseline = model(input, target, train_size)
    output = model(input, changed_target, train_size)

    assert not torch.equal(output[0, 0], baseline[0, 0])
    torch.testing.assert_close(output[0, 1:], baseline[0, 1:])
    torch.testing.assert_close(output[1], baseline[1])


@pytest.mark.parametrize("use_mask", [False, True])
def test_induced_self_attention_block_matches_upstream(
    upstream_tabfm_module: ModuleType,
    use_mask: bool,
) -> None:
    upstream = upstream_tabfm_module.InducedSelfAttentionBlock(
        d_model=16,
        nhead=4,
        dim_ff=32,
        num_inds=5,
    )
    model = InducedSelfAttentionBlock(
        channels=16,
        num_heads=4,
        feedforward_channels=32,
        num_inducing_points=5,
    )
    model.load_state_dict(upstream.state_dict())
    input = torch.randn(3, 7, 16)
    mask = None
    if use_mask:
        mask = torch.rand(3, 1, 1, 7) > 0.25

    torch.testing.assert_close(
        model(input, attn_mask=mask),
        upstream(input, attn_mask=mask),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("use_mask", [False, True])
def test_set_transformer_matches_upstream(
    upstream_tabfm_module: ModuleType,
    use_mask: bool,
) -> None:
    upstream = upstream_tabfm_module.SetTransformer(
        num_blocks=2,
        d_model=16,
        nhead=4,
        dim_ff=32,
        num_inds=5,
    )
    model = SetTransformer(
        num_blocks=2,
        channels=16,
        num_heads=4,
        feedforward_channels=32,
        num_inducing_points=5,
    )
    model.load_state_dict(upstream.state_dict())
    input = torch.randn(3, 7, 16)
    mask = None
    if use_mask:
        mask = torch.rand(3, 1, 1, 7) > 0.25

    torch.testing.assert_close(
        model(input, attn_mask=mask),
        upstream(input, attn_mask=mask),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("col_chunk_size", [None, 2])
def test_col_embedding_matches_upstream(
    upstream_tabfm_module: ModuleType,
    dtype: torch.dtype,
    col_chunk_size: int | None,
) -> None:
    upstream = upstream_tabfm_module.ColEmbedding(
        d_model=16,
        num_blocks=2,
        nhead=4,
        dim_ff=32,
        num_inds=5,
    ).to(dtype)
    model = ColEmbedding(
        channels=16,
        num_blocks=2,
        num_heads=4,
        feedforward_channels=32,
        num_inducing_points=5,
    ).to(dtype)
    model.load_state_dict(upstream.state_dict())
    upstream.col_chunk_size = col_chunk_size
    model.col_chunk_size = col_chunk_size

    input = torch.randn(2, 7, 4, 16, dtype=dtype)
    train_size = torch.tensor([5, 3], dtype=torch.long)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)

    torch.testing.assert_close(
        model(input, train_size),
        upstream(input, train_size),
        rtol=rtol,
        atol=atol,
    )


def test_col_embedding_isolates_query_rows() -> None:
    model = ColEmbedding(
        channels=16,
        num_blocks=2,
        num_heads=4,
        feedforward_channels=32,
        num_inducing_points=5,
    ).eval()
    input = torch.randn(2, 7, 4, 16)
    train_size = torch.tensor([4, 5], dtype=torch.long)
    perturbed = input.clone()
    perturbed[0, 4] += 100

    baseline = model(input, train_size)
    output = model(perturbed, train_size)

    torch.testing.assert_close(output[0, 5:], baseline[0, 5:], rtol=0, atol=0)
    torch.testing.assert_close(output[1], baseline[1], rtol=0, atol=0)
    assert not torch.equal(output[0, 4], baseline[0, 4])


@pytest.mark.parametrize(
    ("train_size", "match"),
    [
        (torch.ones(2, 1, dtype=torch.long), "shape"),
        (torch.ones(2), "integer dtype"),
    ],
)
def test_col_embedding_rejects_invalid_train_size(
    train_size: torch.Tensor,
    match: str,
) -> None:
    model = ColEmbedding(
        channels=16,
        num_blocks=1,
        num_heads=4,
        feedforward_channels=32,
        num_inducing_points=5,
    )

    with pytest.raises(ValueError, match=match):
        model(torch.randn(2, 7, 4, 16), train_size)
