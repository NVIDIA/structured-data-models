import pytest
import torch
from torch import Tensor

from sdm.models.tabfm.core import _TabFM


def _core(
    is_classifier: bool,
    *,
    row_chunk_size: int | None = 4096,
    col_chunk_size: int | None = 16,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> _TabFM:
    return _TabFM(
        embed_dim=2,
        max_classes=3,
        col_num_blocks=1,
        col_num_heads=1,
        col_num_inducing_points=2,
        row_num_blocks=1,
        row_num_heads=1,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_num_heads=2,
        feedforward_factor=2,
        feature_group_size=2,
        num_frequencies=2,
        decoder_hidden_channels=5,
        is_classifier=is_classifier,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        device=device,
        dtype=dtype,
    )


def _inputs(
    is_classifier: bool,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    features = torch.tensor(
        [
            [
                [0.0, float("nan"), 0.5],
                [-0.75, 1.0, -0.25],
                [0.25, -1.0, 0.75],
            ],
            [
                [1.0, -0.5, 99.0],
                [float("nan"), 0.75, -88.0],
                [-1.25, 0.5, 77.0],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    targets = torch.tensor(
        [[1, -100, 99], [2, 0, -100]]
        if is_classifier
        else [[0.5, -100, 99], [1.25, -0.75, -100]],
        device=device,
        dtype=torch.float32 if is_classifier else torch.float64,
    )
    context_size = torch.tensor([1, 2], device=device)
    categorical_mask = torch.tensor(
        [[False, True, False], [True, False, True]],
        device=device,
    )
    active_features = torch.tensor([3, 2], device=device)
    return (
        features,
        targets,
        context_size,
        categorical_mask,
        active_features,
    )


@pytest.mark.parametrize("is_classifier", [True, False])
def test_core_isolates_queries_and_padded_features(
    is_classifier: bool,
) -> None:
    module = _core(is_classifier)
    inputs = _inputs(is_classifier)
    features, targets, context_size, categorical_mask, active_features = inputs
    expected = module(*inputs)

    query = torch.arange(3)[None, :] >= context_size[:, None]
    sentinels = (
        torch.tensor([[-100, 99, -7]], dtype=targets.dtype)
        if is_classifier
        else torch.tensor(
            [[float("nan"), float("inf"), -float("inf")]],
            dtype=targets.dtype,
        )
    ).expand_as(targets)
    changed_targets = torch.where(query, sentinels, targets)
    torch.testing.assert_close(
        module(
            features,
            changed_targets,
            context_size,
            categorical_mask,
            active_features,
        ),
        expected,
        rtol=0,
        atol=0,
    )

    changed_features = features.clone()
    changed_features[1, :, 2].add_(1000)
    torch.testing.assert_close(
        module(
            changed_features,
            targets,
            context_size,
            categorical_mask,
            active_features,
        ),
        expected,
    )


def test_core_rejects_complex_features() -> None:
    features, targets, context_size, categorical_mask, active_features = (
        _inputs(False)
    )

    with pytest.raises(ValueError, match="complex"):
        _core(False)(
            features.to(torch.complex64),
            targets,
            context_size,
            categorical_mask,
            active_features,
        )
