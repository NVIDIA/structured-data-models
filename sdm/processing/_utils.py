import torch
from torch import Tensor


def _draw_device(
    generator: torch.Generator | None,
    default: torch.device | None = None,
) -> torch.device | None:
    """Return the device random draws must be placed on.

    Draws with a generator must live on the generator's device; without
    one, ``default`` selects the global generator to draw from.
    """
    if generator is None:
        return default
    return generator.device


def _as_float(inp: Tensor) -> Tensor:
    """Return a floating-point tensor for numeric transforms.

    Floating-point inputs are returned unchanged; integer inputs are promoted
    to the default floating dtype.
    """
    if inp.is_floating_point():
        return inp
    return inp.to(dtype=torch.get_default_dtype())
