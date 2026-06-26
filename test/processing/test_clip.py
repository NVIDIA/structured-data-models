import pytest
import torch
from sdm.processing import Clip


def test_clip_quantile_bounds_and_transform() -> None:
    input = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [100.0, 40.0],
        ]
    )

    processor = Clip(q_low=0.25, q_high=0.75).fit(input)
    expected_bounds = torch.quantile(input, torch.tensor([0.25, 0.75]), dim=0)
    expected = input.clamp(min=expected_bounds[0], max=expected_bounds[1])

    assert torch.allclose(processor.lower_bound, expected_bounds[0])
    assert torch.allclose(processor.upper_bound, expected_bounds[1])
    assert torch.equal(processor.transform(input), expected)
    assert torch.equal(processor.inverse_transform(expected), expected)


def test_clip_default_uses_min_max_bounds() -> None:
    input = torch.tensor([[0.0, 5.0], [2.0, 7.0], [4.0, 9.0]])

    processor = Clip().fit(input)

    assert torch.equal(processor.lower_bound, torch.tensor([0.0, 5.0]))
    assert torch.equal(processor.upper_bound, torch.tensor([4.0, 9.0]))
    assert torch.equal(processor.transform(input), input)


def test_clip_constant_columns_are_exact() -> None:
    input = torch.full((4, 2), 3.0)

    processor = Clip(q_low=0.02, q_high=0.98).fit(input)

    assert torch.equal(processor.lower_bound, torch.full((2,), 3.0))
    assert torch.equal(processor.upper_bound, torch.full((2,), 3.0))
    assert torch.equal(processor.transform(input), input)


def test_clip_nan_columns_follow_torch_quantile() -> None:
    input = torch.tensor(
        [
            [0.0, 1.0],
            [torch.nan, 2.0],
            [4.0, 3.0],
        ]
    )

    processor = Clip(q_low=0.0, q_high=1.0).fit(input)
    transformed = processor.transform(input)

    assert torch.isnan(processor.lower_bound[0])
    assert torch.isnan(processor.upper_bound[0])
    assert torch.isnan(transformed[:, 0]).all()
    assert torch.equal(transformed[:, 1], input[:, 1])


def test_clip_rejects_invalid_quantiles_with_parameter_names() -> None:
    with pytest.raises(ValueError, match="q_low <= q_high"):
        Clip(q_low=0.75, q_high=0.25)
