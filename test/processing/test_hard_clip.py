import pytest
import torch
from sdm import TableTensor
from sdm.processing import HardClip
from sdm.testing import withCUDA


@withCUDA
def test_hard_clip_clamps_fixed_bounds_and_preserves_metadata(
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [[-101.0, -100.0], [100.0, 101.0]],
            device=device,
        ),
        columns=("a", "b"),
    )

    actual = HardClip(min_value=-100.0, max_value=100.0).transform(table)

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor(
            [[-100.0, -100.0], [100.0, 100.0]],
            device=device,
        ),
    )
    assert actual.columns == table.columns
    assert actual.device == table.device
    assert repr(HardClip(min_value=-100.0, max_value=100.0)) == (
        "HardClip(min_value=-100.0, max_value=100.0)"
    )


def test_hard_clip_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="min_value"):
        HardClip(min_value=1.0, max_value=-1.0)
