import pytest
import torch

from sdm.models.tabfm.embedding import _CellEmbedder
from sdm.testing import withCUDA


def _set_golden_state(module: _CellEmbedder) -> None:
    with torch.no_grad():
        module.fourier_frequencies.copy_(
            torch.tensor([[0.25, -0.5], [0.75, 1.0], [-1.25, 0.5]])
        )
        module.fourier_frequencies_cat.copy_(
            torch.tensor([[-0.3, 0.6], [0.9, -1.2], [0.4, 0.8]])
        )
        module.in_linear.weight.copy_(torch.arange(12).view(3, 4) / 10 - 0.5)
        module.in_linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        module.in_linear_cat.weight.copy_(
            torch.arange(12).view(3, 4).flip(1) / 8 - 0.4
        )
        module.in_linear_cat.bias.copy_(torch.tensor([-0.1, 0.2, -0.3]))


def test_cell_embedder_matches_google_golden() -> None:
    module = _CellEmbedder(
        embed_dim=3,
        feature_group_size=3,
        num_frequencies=2,
        row_chunk_size=None,
    )
    _set_golden_state(module)
    features = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
            [[-1.0, 0.5, 2.5, 9.0], [3.5, -2.0, 8.0, 4.0]],
        ]
    )
    categorical_mask = torch.tensor(
        [[False, True, False, True], [True, False, True, False]]
    )

    output = module(
        features,
        categorical_mask=categorical_mask,
        active_features=torch.tensor([4, 3]),
    )

    # Frozen from google-research/tabfm@b8a8b090 CellEmbedder._cell and
    # forward's active-feature mask, using the state and inputs above.
    expected = torch.tensor(
        [
            [
                [
                    [-0.72497445, 1.40888643, 2.74274731],
                    [-1.87061846, 0.27883190, 3.22828221],
                    [-0.49351242, 1.12562966, 1.94477177],
                    [-0.51162553, 0.27370167, 1.85902917],
                ],
                [
                    [-0.54425102, -0.42755699, -1.11086297],
                    [0.10699922, -0.58954960, -0.48609853],
                    [0.16168760, -0.20072001, -1.36312771],
                    [1.75370097, -1.39106739, -3.73583579],
                ],
            ],
            [
                [
                    [-1.85975420, 0.51303262, 2.08581948],
                    [-0.02784103, 0.49377781, 1.81539655],
                    [-1.23775697, 1.72518611, 2.28812909],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [1.03704882, 0.70451480, -0.42801923],
                    [-0.01798820, 0.48827714, 1.79454279],
                    [0.23589724, -0.31185949, -3.25961637],
                    [0.0, 0.0, 0.0],
                ],
            ],
        ]
    )
    torch.testing.assert_close(output, expected)


def test_cell_embedder_ignores_padded_features() -> None:
    generator = torch.Generator().manual_seed(0)
    module = _CellEmbedder(embed_dim=4, num_frequencies=2)
    with torch.no_grad():
        module.fourier_frequencies.normal_(generator=generator)
        module.fourier_frequencies_cat.normal_(generator=generator)
    features = torch.randn(2, 3, 5, generator=generator)
    categorical_mask = torch.tensor(
        [[False, True, False, True, False], [True, False, True, False, True]]
    )
    active_features = torch.tensor([5, 2])

    expected = module(features, categorical_mask, active_features)
    perturbed = features.clone()
    perturbed[1, :, 2:] = 1000 * torch.randn(
        perturbed[1, :, 2:].shape,
        generator=generator,
    )
    perturbed_mask = categorical_mask.clone()
    perturbed_mask[1, 2:] = ~perturbed_mask[1, 2:]
    output = module(perturbed, perturbed_mask, active_features)

    torch.testing.assert_close(output[1, :, :2], expected[1, :, :2])
    assert torch.count_nonzero(output[1, :, 2:]) == 0


def test_cell_embedder_handles_empty_active_prefixes() -> None:
    module = _CellEmbedder(embed_dim=4, num_frequencies=2)

    output = module(
        torch.randn(2, 3, 4),
        active_features=torch.tensor([0, 4]),
    )

    assert torch.count_nonzero(output[0]) == 0


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cell_embedder_chunking_dtype_and_device(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device=device).manual_seed(0)
    module = _CellEmbedder(
        embed_dim=4,
        num_frequencies=2,
        row_chunk_size=None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        module.fourier_frequencies.normal_(generator=generator)
        module.fourier_frequencies_cat.normal_(generator=generator)
    features = torch.randn(
        2,
        5,
        4,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    categorical_mask = torch.tensor(
        [[False, True, False, True], [True, False, True, False]],
        device=device,
    )
    active_features = torch.tensor([4, 3], device=device)

    expected = module(features, categorical_mask, active_features)
    module.row_chunk_size = 2
    output = module(features, categorical_mask, active_features)

    torch.testing.assert_close(output, expected)
    assert output.dtype == dtype
    assert output.device == device
    assert module.fourier_frequencies.dtype == torch.float32
    assert module.fourier_frequencies_cat.dtype == torch.float32


def test_cell_embedder_compiles_fullgraph() -> None:
    module = _CellEmbedder(embed_dim=4, num_frequencies=2, row_chunk_size=2)
    features = torch.randn(2, 3, 4)
    categorical_mask = torch.tensor(
        [[False, True, False, True], [True, False, True, False]]
    )
    active_features = torch.tensor([4, 3])
    expected = module(features, categorical_mask, active_features)

    compiled = torch.compile(module, backend="eager", fullgraph=True)

    torch.testing.assert_close(
        compiled(features, categorical_mask, active_features),
        expected,
    )
