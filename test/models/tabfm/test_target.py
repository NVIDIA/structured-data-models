import pytest
import torch
from sdm.models.tabfm.target import OneHotAndLinear
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_one_hot_and_linear_matches_tabfm_behavior(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = OneHotAndLinear(
        num_classes=3,
        channels=2,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        module.projection.weight.copy_(
            torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                device=device,
                dtype=dtype,
            )
        )
        module.projection.bias.copy_(
            torch.tensor([0.5, -0.5], device=device, dtype=dtype)
        )
    target = torch.tensor([[-1, 0, 2, 3]], device=device)

    output = module(target)

    expected = torch.tensor(
        [
            [
                [0.5, -0.5],
                [1.5, 3.5],
                [3.5, 5.5],
                [0.5, -0.5],
            ]
        ],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == (*target.shape, 2)
    assert output.device == device
    assert output.dtype == dtype


def test_one_hot_and_linear_preserves_checkpoint_names() -> None:
    module = OneHotAndLinear(num_classes=3, channels=2)

    assert set(module.state_dict()) == {
        "projection.weight",
        "projection.bias",
    }

    restored = OneHotAndLinear(num_classes=3, channels=2)
    restored.load_state_dict(module.state_dict(), strict=True)


@pytest.mark.parametrize(
    ("num_classes", "channels", "match"),
    [(0, 2, "num_classes"), (3, 0, "channels")],
)
def test_one_hot_and_linear_rejects_invalid_configuration(
    num_classes: int,
    channels: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        OneHotAndLinear(num_classes=num_classes, channels=channels)
