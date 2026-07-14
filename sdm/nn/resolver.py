from typing import Any

import torch
from torch.nn import Module


def _normalize_string(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def normalization_resolver(
    query: torch.nn.Module | str,
    *args: Any,
    **kwargs: Any,
) -> Module:
    """Resolve a public PyTorch normalization name to a module.

    Names are case-insensitive and may contain hyphens, underscores, or spaces.
    The ``"norm"`` suffix is optional.

    Args:
        query: The normalization name or an existing module to return
            unchanged.
        *args: Additional positional arguments passed to the normalization
            layer constructor.
        **kwargs: Additional keyword arguments passed to the normalization
            layer constructor.
    """
    if isinstance(query, Module):
        return query

    modules: tuple[type[Module], ...] = tuple(
        value
        for value in vars(torch.nn).values()
        if (
            isinstance(value, type)
            and issubclass(value, torch.nn.Module)
            and "norm" in value.__module__.rsplit(".", maxsplit=1)[-1]
        )
    )

    query_repr = _normalize_string(query)
    for cls in modules:
        cls_repr = _normalize_string(cls.__name__)
        if query_repr in {cls_repr, cls_repr.replace("norm", "")}:
            return cls(*args, **kwargs)

    raise ValueError(
        f"Could not resolve normalization '{query}'. "
        f"Available choices: {', '.join(cls.__name__ for cls in modules)}"
    )
