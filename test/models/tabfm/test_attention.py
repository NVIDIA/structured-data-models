import pytest
import torch
from sdm.models.tabfm.attention import RMSNorm
from torch import Tensor


def reference_rms_norm(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    input_dtype = input.dtype
    input_float = input.float()
    variance = input_float.square().mean(dim=-1, keepdim=True)
    normalized = input_float * (variance + epsilon).rsqrt()
    return (normalized * weight.float()).to(input_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_reference(dtype: torch.dtype) -> None:
    model = RMSNorm(channels=8, epsilon=1e-6, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.randn_like(model.weight))
    input = torch.randn(2, 5, 8, dtype=dtype)

    output = model(input)
    expected = reference_rms_norm(
        input=input,
        weight=model.weight,
        epsilon=model.epsilon,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert output.device == input.device


def test_rms_norm_state_dict_matches_checkpoint_name() -> None:
    model = RMSNorm(channels=8)

    assert set(model.state_dict()) == {"weight"}
    assert model.state_dict()["weight"].shape == (8,)


@pytest.mark.parametrize(
    ("channels", "epsilon", "message"),
    [
        (0, 1e-6, "channels must be positive"),
        (8, -1.0, "epsilon must be non-negative"),
    ],
)
def test_rms_norm_rejects_invalid_configuration(
    channels: int,
    epsilon: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RMSNorm(channels=channels, epsilon=epsilon)
