from torch import Tensor


def _as_float(inp: Tensor) -> Tensor:
    """Return a floating-point tensor for numeric transforms.

    Floating-point inputs are returned unchanged; integer inputs are promoted
    to the default floating dtype.
    """
    if inp.is_floating_point():
        return inp
    return inp.to(dtype=torch.get_default_dtype())
