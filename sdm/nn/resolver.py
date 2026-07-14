from typing import Any

import torch
from torch.nn import Module


def _normalize_string(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def normalization_resolver(
    query: Module | str,
    *args: Any,
    **kwargs: Any,
) -> Module:
    """Resolve a public PyTorch normalization name to a module.

    Names are case-insensitive and may contain hyphens, underscores, or spaces.
    The ``Norm`` suffix is optional.

    Args:
        query: The normalization name or an existing module to return
            unchanged.
        *args: Positional arguments passed to the normalization constructor.
        **kwargs: Keyword arguments passed to the normalization constructor.

    Returns:
        The existing module or a newly constructed normalization module.

    Raises:
        TypeError: If ``query`` is neither a string nor a module.
        ValueError: If the normalization name is unknown.
    """
    if isinstance(query, Module):
        return query
    if not isinstance(query, str):
        raise TypeError("`query` must be a string or torch.nn.Module")

    normalization_classes: tuple[type[Module], ...] = tuple(
        value
        for value in vars(torch.nn).values()
        if (
            isinstance(value, type)
            and issubclass(value, Module)
            and "norm" in value.__module__.rsplit(".", maxsplit=1)[-1]
        )
    )
    query_repr = _normalize_string(query)
    for normalization_cls in normalization_classes:
        cls_repr = _normalize_string(normalization_cls.__name__)
        if query_repr in {cls_repr, cls_repr.replace("norm", "")}:
            return normalization_cls(*args, **kwargs)

    choices = ", ".join(cls.__name__ for cls in normalization_classes)
    raise ValueError(
        f"Could not resolve normalization '{query}'. "
        f"Available choices: {choices}"
    )
