import torch
from torch import Tensor


def frozen_google_state_dict(
    module: torch.nn.Module,
    *,
    rope_value: float = 0.75,
) -> dict[str, Tensor]:
    """Create deterministic state for pinned Google TabFM parity tests."""
    state = {}
    for name, tensor in module.state_dict().items():
        google_name = name
        if name.endswith("rope.inv_freq"):
            google_name = name.removesuffix("inv_freq") + "freqs"
            values = torch.full(
                (tensor.numel(),),
                rope_value,
                device=tensor.device,
                dtype=torch.float32,
            )
        elif google_name.endswith("_ln.weight"):
            offset = (sum(google_name.encode()) % 7) * 0.05
            values = torch.linspace(
                0.8,
                1.2,
                tensor.numel(),
                device=tensor.device,
            ).add_(offset)
        elif google_name.endswith("per_dim_scale"):
            values = torch.linspace(
                -0.3,
                0.2,
                tensor.numel(),
                device=tensor.device,
            )
        else:
            offset = (sum(google_name.encode()) % 11 - 5) * 0.003
            start, end = (
                (-0.04, 0.04)
                if google_name.endswith(".bias")
                else (-0.15, 0.15)
            )
            values = torch.linspace(
                start,
                end,
                tensor.numel(),
                device=tensor.device,
            ).add_(offset)
        state[name] = values.reshape(tensor.shape)
    return state
