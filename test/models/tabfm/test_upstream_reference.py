from types import ModuleType

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    ("is_classifier", "output_channels"),
    [(True, 3), (False, 1)],
)
def test_upstream_tiny_forward_is_deterministic(
    upstream_tabfm_module: ModuleType,
    is_classifier: bool,
    output_channels: int,
) -> None:
    """Run the forward contract used by later SDM parity tests."""
    torch.manual_seed(0)
    model = upstream_tabfm_module.TabFM(
        embed_dim=8,
        max_classes=3,
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
    ).eval()

    batch_size, num_rows, num_features = 2, 6, 4
    x = torch.randn(batch_size, num_rows, num_features)
    if is_classifier:
        y = torch.randint(0, 3, (batch_size, num_rows))
    else:
        y = torch.randn(batch_size, num_rows)

    train_size = torch.tensor([4, 3], dtype=torch.long)
    cat_mask = torch.tensor(
        [[False, True, False, True], [True, False, False, False]]
    )
    active_features = torch.tensor([4, 3], dtype=torch.long)

    with torch.inference_mode():
        first = model(
            x,
            y,
            train_size,
            cat_mask=cat_mask,
            d=active_features,
        )
        second = model(
            x,
            y,
            train_size,
            cat_mask=cat_mask,
            d=active_features,
        )

    assert first.shape == (batch_size, num_rows, output_channels)
    assert first.isfinite().all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
