from types import ModuleType

import pytest
import torch
from sdm.models.tabfm.row_interaction import RowInteraction


@pytest.mark.parametrize("output_full", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_active_features", [False, True])
@pytest.mark.parametrize("row_chunk_size", [None, 3])
def test_row_interaction_matches_upstream(
    upstream_tabfm_module: ModuleType,
    output_full: bool,
    dtype: torch.dtype,
    use_active_features: bool,
    row_chunk_size: int | None,
) -> None:
    upstream = upstream_tabfm_module.RowInteraction(
        d_model=16,
        num_blocks=2,
        nhead=4,
        dim_ff=32,
        num_cls=2,
        rope_base=100_000.0,
        output_full=output_full,
    ).to(dtype)
    model = RowInteraction(
        channels=16,
        num_blocks=2,
        num_heads=4,
        feedforward_channels=32,
        num_cls=2,
        rope_theta=100_000.0,
        output_full=output_full,
    ).to(dtype)
    model.load_state_dict(upstream.state_dict())
    upstream.row_chunk_size = row_chunk_size
    model.row_chunk_size = row_chunk_size

    input = torch.randn(2, 5, 7, 16, dtype=dtype)
    active_features = None
    if use_active_features:
        active_features = torch.tensor([5, 3], dtype=torch.long)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)

    output = model(input, d=active_features)
    expected = upstream(input, d=active_features)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    if output_full:
        assert output.shape == (2, 5, 7, 16)
    else:
        assert output.shape == (2, 5, 32)


@pytest.mark.parametrize("output_full", [False, True])
def test_row_interaction_masks_padded_feature_tokens(
    output_full: bool,
) -> None:
    model = RowInteraction(
        channels=16,
        num_blocks=2,
        num_heads=4,
        feedforward_channels=32,
        num_cls=2,
        output_full=output_full,
    ).eval()
    input = torch.randn(1, 4, 7, 16)
    active_features = torch.tensor([3], dtype=torch.long)
    perturbed = input.clone()
    perturbed[:, :, 5:] += 100

    baseline = model(input, d=active_features)
    output = model(perturbed, d=active_features)

    if output_full:
        torch.testing.assert_close(
            output[:, :, :5],
            baseline[:, :, :5],
            rtol=0,
            atol=0,
        )
        assert not torch.equal(output[:, :, 5:], baseline[:, :, 5:])
    else:
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("output_full", [False, True])
def test_row_interaction_processes_rows_independently(
    output_full: bool,
) -> None:
    model = RowInteraction(
        channels=16,
        num_blocks=2,
        num_heads=4,
        feedforward_channels=32,
        num_cls=2,
        output_full=output_full,
    ).eval()
    input = torch.randn(2, 5, 7, 16)
    perturbed = input.clone()
    perturbed[0, 0] += 100

    baseline = model(input)
    output = model(perturbed)

    assert not torch.equal(output[0, 0], baseline[0, 0])
    torch.testing.assert_close(output[0, 1:], baseline[0, 1:])
    torch.testing.assert_close(output[1], baseline[1])


@pytest.mark.parametrize(
    ("active_features", "match"),
    [
        (torch.ones(2, 1, dtype=torch.long), "shape"),
        (torch.ones(2), "integer dtype"),
    ],
)
def test_row_interaction_rejects_invalid_active_features(
    active_features: torch.Tensor,
    match: str,
) -> None:
    model = RowInteraction(
        channels=16,
        num_blocks=1,
        num_heads=4,
        feedforward_channels=32,
        num_cls=2,
    )

    with pytest.raises(ValueError, match=match):
        model(torch.randn(2, 5, 7, 16), d=active_features)
