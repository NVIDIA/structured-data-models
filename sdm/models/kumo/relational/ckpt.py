import torch
from torch import Tensor


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D103
    legacy_keys = tuple(
        f"gnn.{name}_lin.weight"
        for name in ("sum", "avg", "std", "min", "max")
    )
    if not all(key in ckpt for key in legacy_keys):
        return ckpt

    out: dict[str, Tensor] = {}
    for key, value in ckpt.items():
        if key not in legacy_keys:
            out[key] = value
    out["gnn.aggregation_lin.weight"] = torch.cat(
        [ckpt[key] for key in legacy_keys], dim=1
    )
    return out
