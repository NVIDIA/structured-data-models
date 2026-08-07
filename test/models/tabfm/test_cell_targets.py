import pytest
import torch

from sdm.models.tabfm.embedding import _CellEmbedder
from sdm.testing import withCUDA


def _load_golden_state(
    module: _CellEmbedder,
    is_classifier: bool,
) -> None:
    state = module.state_dict()
    state["in_linear.weight"] = torch.zeros_like(state["in_linear.weight"])
    state["in_linear.bias"] = torch.zeros_like(state["in_linear.bias"])
    values = (
        {
            "y_embedder_lookup.weight": [
                [-0.5],
                [0.25],
                [0.75],
                [-1.0],
            ]
        }
        if is_classifier
        else {
            "y_embedder_lookup.layers.0.weight": [
                [-0.5],
                [-0.25],
                [0.0],
                [0.25],
                [0.5],
                [0.75],
            ],
            "y_embedder_lookup.layers.0.bias": [
                -0.25,
                -0.125,
                0.0,
                0.125,
                0.25,
                0.375,
            ],
            "y_embedder_lookup.layers.1.weight": [
                [-0.625, -0.375, -0.125, 0.125, 0.375, 0.625]
            ],
            "y_embedder_lookup.layers.1.bias": [0.25],
        }
    )
    for key, value in values.items():
        state[key] = torch.tensor(
            value,
            device=state[key].device,
            dtype=state[key].dtype,
        )
    module.load_state_dict(state)


@withCUDA
@pytest.mark.parametrize(
    ("is_classifier", "dtype", "expected"),
    [
        (
            True,
            torch.float32,
            [-0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 0.75, 0.0],
        ),
        (
            True,
            torch.bfloat16,
            [-0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 0.75, 0.0],
        ),
        (
            False,
            torch.float32,
            [
                0.8948780298,
                0.8948780298,
                0.0,
                0.0,
                1.4532704353,
                0.0,
                0.1200520992,
                0.0,
            ],
        ),
        (
            False,
            torch.bfloat16,
            [
                0.89453125,
                0.89453125,
                0.0,
                0.0,
                1.453125,
                0.0,
                0.1201171875,
                0.0,
            ],
        ),
    ],
)
def test_cell_targets_match_google_golden(
    device: torch.device,
    is_classifier: bool,
    dtype: torch.dtype,
    expected: list[float],
) -> None:
    module = _CellEmbedder(
        embed_dim=1,
        feature_group_size=1,
        num_frequencies=1,
        row_chunk_size=1,
        device=device,
        dtype=dtype,
        is_classifier=is_classifier,
        max_classes=4,
    )
    _load_golden_state(module, is_classifier)
    features = torch.tensor(
        [[[0.0, 0.5], [-0.5, 1.0]], [[1.0, -1.0], [0.25, 0.75]]],
        device=device,
        dtype=dtype,
    )
    targets = torch.tensor(
        [[-100.0, 8.0], [5.0, 2.0]]
        if is_classifier
        else [[0.5, 99.0], [1.25, -0.75]],
        device=device,
        dtype=torch.float32 if is_classifier else torch.float64,
    )

    output = module(
        features,
        active_features=torch.tensor([2, 1], device=device),
        targets=targets,
        context_size=torch.tensor([1, 2], device=device),
    )

    # Frozen from google-research/tabfm@b8a8b090's PyTorch CellEmbedder.
    torch.testing.assert_close(
        output,
        torch.tensor(expected, device=device, dtype=dtype).view(2, 2, 2, 1),
    )


@pytest.mark.parametrize("is_classifier", [True, False])
def test_cell_targets_do_not_leak_into_query_or_padding(
    is_classifier: bool,
) -> None:
    module = _CellEmbedder(
        embed_dim=3,
        num_frequencies=2,
        is_classifier=is_classifier,
        max_classes=4,
    )
    features = torch.arange(36, dtype=torch.float32).view(3, 3, 4) / 10
    context_size = torch.tensor([0, 2, 3])
    active_features = torch.tensor([4, 2, 0])
    targets = (
        torch.tensor([[0, 1, 2], [2, 1, 0], [1, 2, 0]])
        if is_classifier
        else torch.tensor(
            [[0.25, -0.5, 1.0], [2.0, -1.0, 0.5], [1.5, 0.0, -2.0]]
        )
    )
    query = torch.arange(3)[None, :] >= context_size[:, None]
    sentinels = (
        torch.tensor([[-100, 100, -100]]).expand_as(targets)
        if is_classifier
        else torch.tensor([[float("nan"), 1e6, float("nan")]]).expand_as(
            targets
        )
    )
    changed_targets = torch.where(query, sentinels, targets)

    expected = module(
        features,
        active_features=active_features,
        targets=targets,
        context_size=context_size,
    )
    output = module(
        features,
        active_features=active_features,
        targets=changed_targets,
        context_size=context_size,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.isfinite().all()
    assert torch.count_nonzero(output[1, :, 2:]) == 0
    assert torch.count_nonzero(output[2]) == 0


def test_cell_targets_are_required_for_task_embedder() -> None:
    module = _CellEmbedder(embed_dim=2, is_classifier=False)

    with pytest.raises(ValueError, match="targets"):
        module(torch.zeros(1, 2, 2))
