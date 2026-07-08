import pytest
import torch
from sdm import TableTensor
from sdm.processing import Clamp


@pytest.mark.parametrize(
    ("min_value", "max_value", "expected"),
    [
        (-1.0, 2.0, [-1.0, -1.0, 0.0, 2.0]),
        (0.0, None, [0.0, 0.0, 0.0, 3.0]),
        (None, 1.0, [-3.0, -1.0, 0.0, 1.0]),
    ],
)
def test_clamp_fixed_bounds(
    min_value: float | None,
    max_value: float | None,
    expected: list[float],
) -> None:
    processor = Clamp(min_value=min_value, max_value=max_value)
    data = torch.tensor(
        [[-3.0], [-1.0], [0.0], [3.0]],
        dtype=torch.bfloat16,
    )
    input = TableTensor(
        columns={"numerical": ["value"]},
        numerical=data,
    )

    output = processor.transform(input)

    torch.testing.assert_close(
        output.numerical.squeeze(-1),
        torch.tensor(expected, dtype=data.dtype),
    )
    assert output.numerical.dtype == input.numerical.dtype
    assert output.device == input.device


@pytest.mark.parametrize(
    ("min_value", "max_value", "match"),
    [
        (None, None, "at least one"),
        (2.0, 1.0, "must not exceed"),
    ],
)
def test_clamp_rejects_invalid_bounds(
    min_value: float | None,
    max_value: float | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Clamp(min_value=min_value, max_value=max_value)
