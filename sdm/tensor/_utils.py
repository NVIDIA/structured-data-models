from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import Tensor

aten = torch.ops.aten

_VIEW_FUNCTIONS = frozenset(
    {
        aten.alias.default,
        aten.detach.default,
        aten.view.default,
        aten._unsafe_view.default,
        aten.squeeze.default,
        aten.squeeze.dim,
        aten.squeeze.dims,
        aten.unsqueeze.default,
        aten.expand.default,
        aten.transpose.int,
        aten.permute.default,
        aten.select.int,
        aten.slice.Tensor,
        aten.narrow.default,
        aten.unbind.int,
        aten.split.Tensor,
        aten.split.sizes,
        aten.split.default,
        aten.split_with_sizes.default,
    }
)


@contextlib.contextmanager
def _preserve_view_inference_mode(
    func: Callable[..., Any],
    input: Tensor,
) -> Iterator[None]:
    """Preserve an outer tensor's inference state when rebuilding a view."""
    if func in _VIEW_FUNCTIONS:
        with torch.inference_mode(input.is_inference()):
            yield
        return

    yield
