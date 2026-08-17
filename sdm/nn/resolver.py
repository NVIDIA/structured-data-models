from collections.abc import Callable
from typing import Any

import torch


def _normalize_string(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def normalization_resolver(
    norm: str | Callable[..., torch.nn.Module],
    *args: Any,
    **kwargs: Any,
) -> torch.nn.Module:
    """Resolve a normalization layer name or callable to a module.

    Names are matched against the public PyTorch normalization layers,
    case-insensitively; hyphens, underscores, and spaces are ignored and the
    ``"norm"`` suffix is optional. A callable (*e.g.*, a layer class) is
    called with the given arguments, so every resolution yields a fresh
    module. A module instance is returned unchanged and is therefore shared
    across resolution sites.

    Args:
        norm: The normalization layer name, a callable returning the
            normalization layer.
        *args: Additional positional arguments passed to the normalization
            layer constructor.
        **kwargs: Additional keyword arguments passed to the normalization
            layer constructor.
    """
    if not isinstance(norm, str):
        return norm(*args, **kwargs)

    modules: tuple[type[torch.nn.Module], ...] = tuple(
        value
        for value in vars(torch.nn).values()
        if (
            isinstance(value, type)
            and issubclass(value, torch.nn.Module)
            and "norm" in value.__module__.rsplit(".", maxsplit=1)[-1]
        )
    )

    norm_repr = _normalize_string(norm)
    for cls in modules:
        cls_repr = _normalize_string(cls.__name__)
        if norm_repr in {cls_repr, cls_repr.replace("norm", "")}:
            return cls(*args, **kwargs)

    raise ValueError(
        f"Could not resolve normalization {norm!r}. "
        f"Available choices: {', '.join(cls.__name__ for cls in modules)}"
    )
