import torch

from sdm import TableTensor
from sdm.processing import ReplaceInf
from sdm.testing import withCUDA


@withCUDA
def test_replace_inf_converts_infinities_to_nan(device: torch.device) -> None:
    inp = torch.tensor(
        [[1.0, float("inf")], [float("-inf"), 2.0]],
        device=device,
    )

    out = ReplaceInf().transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(
        out.isnan(),
        torch.tensor([[False, True], [True, False]], device=device),
    )
    assert torch.equal(
        out.nan_to_num(),
        torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device),
    )
