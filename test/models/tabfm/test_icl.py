from types import ModuleType

import pytest
import torch
from sdm.models.tabfm.icl import ICLearning, OneHotAndLinear


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_one_hot_and_linear_matches_upstream(
    upstream_tabfm_module: ModuleType,
    dtype: torch.dtype,
) -> None:
    upstream = upstream_tabfm_module.OneHotAndLinear(
        num_classes=3,
        embed_dim=8,
    ).to(dtype)
    model = OneHotAndLinear(num_classes=3, channels=8).to(dtype)
    model.load_state_dict(upstream.state_dict())
    target = torch.tensor([[0, 1, 2, -1, 3, 99]])
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)

    output = model(target)
    torch.testing.assert_close(
        output,
        upstream(target),
        rtol=rtol,
        atol=atol,
    )
    expected_invalid = model.projection.bias.expand(3, -1)
    torch.testing.assert_close(output[0, 3:], expected_invalid)


@pytest.mark.parametrize("is_classifier", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_icl_matches_upstream(
    upstream_tabfm_module: ModuleType,
    is_classifier: bool,
    dtype: torch.dtype,
) -> None:
    upstream = upstream_tabfm_module.ICLearning(
        d_model=16,
        num_blocks=2,
        nhead=4,
        max_classes=3,
        dim_ff=32,
        decoder_hidden=24,
        is_classifier=is_classifier,
    ).to(dtype)
    model = ICLearning(
        channels=16,
        num_blocks=2,
        num_heads=4,
        max_classes=3,
        feedforward_channels=32,
        decoder_hidden=24,
        is_classifier=is_classifier,
    ).to(dtype)
    model.load_state_dict(upstream.state_dict())

    input = torch.randn(2, 7, 16, dtype=dtype)
    if is_classifier:
        target = torch.randint(0, 3, (2, 7))
        target[0, 5:] = -100
        target[1, 4:] = -100
        output_channels = 3
    else:
        target = torch.randn(2, 7, dtype=dtype)
        output_channels = 1
    train_size = torch.tensor([5, 4], dtype=torch.long)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)

    output = model(input, target, train_size)
    expected = upstream(input, target, train_size)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    assert output.shape == (2, 7, output_channels)


@pytest.mark.parametrize("is_classifier", [False, True])
def test_icl_ignores_query_targets(is_classifier: bool) -> None:
    model = ICLearning(
        channels=16,
        num_blocks=2,
        num_heads=4,
        max_classes=3,
        feedforward_channels=32,
        decoder_hidden=24,
        is_classifier=is_classifier,
    ).eval()
    input = torch.randn(2, 7, 16)
    if is_classifier:
        target = torch.randint(0, 3, (2, 7))
        changed_query_target = (target + 1) % 3
    else:
        target = torch.randn(2, 7)
        changed_query_target = target + 100
    train_size = torch.tensor([5, 4], dtype=torch.long)
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


@pytest.mark.parametrize("is_classifier", [False, True])
def test_icl_context_targets_affect_query_predictions(
    is_classifier: bool,
) -> None:
    model = ICLearning(
        channels=16,
        num_blocks=2,
        num_heads=4,
        max_classes=3,
        feedforward_channels=32,
        decoder_hidden=24,
        is_classifier=is_classifier,
    ).eval()
    input = torch.randn(2, 7, 16)
    if is_classifier:
        target = torch.randint(0, 3, (2, 7))
        changed_target = target.clone()
        changed_target[0, 0] = (changed_target[0, 0] + 1) % 3
    else:
        target = torch.randn(2, 7)
        changed_target = target.clone()
        changed_target[0, 0] += 100
    train_size = torch.tensor([5, 4], dtype=torch.long)

    baseline = model(input, target, train_size)
    output = model(input, changed_target, train_size)

    assert not torch.equal(output[0, 5:], baseline[0, 5:])
    torch.testing.assert_close(output[1], baseline[1])


def test_icl_isolates_query_rows() -> None:
    model = ICLearning(
        channels=16,
        num_blocks=2,
        num_heads=4,
        max_classes=3,
        feedforward_channels=32,
        decoder_hidden=24,
    ).eval()
    input = torch.randn(2, 7, 16)
    target = torch.randint(0, 3, (2, 7))
    train_size = torch.tensor([4, 5], dtype=torch.long)
    perturbed = input.clone()
    perturbed[0, 4] += 100

    baseline = model(input, target, train_size)
    output = model(perturbed, target, train_size)

    assert not torch.equal(output[0, 4], baseline[0, 4])
    torch.testing.assert_close(output[0, 5:], baseline[0, 5:])
    torch.testing.assert_close(output[1], baseline[1])


@pytest.mark.parametrize(
    ("train_size", "match"),
    [
        (torch.ones(2, 1, dtype=torch.long), "shape"),
        (torch.ones(2), "integer dtype"),
    ],
)
def test_icl_rejects_invalid_train_size(
    train_size: torch.Tensor,
    match: str,
) -> None:
    model = ICLearning(
        channels=16,
        num_blocks=1,
        num_heads=4,
        max_classes=3,
        feedforward_channels=32,
        decoder_hidden=24,
    )

    with pytest.raises(ValueError, match=match):
        model(
            torch.randn(2, 7, 16),
            torch.randint(0, 3, (2, 7)),
            train_size,
        )
