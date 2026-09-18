# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

_attention_quantization: ContextVar[Literal["fp8"] | None] = ContextVar(
    "sdm_attention_quantization", default=None
)


@contextmanager
def optimize(
    *, attention: Literal["fp8", "inherit"] | None = "inherit"
) -> Iterator[None]:
    """Temporarily configure execution optimizations in the current context.

    Args:
        attention: ``"fp8"`` enables quantization in eligible ICL attention
            layers during supported CUDA inference with more than 8192 context
            rows. ``None`` disables it; ``"inherit"`` preserves the enclosing
            setting. Weights and the final ICL layer remain unquantized.

    Yields:
        None. Previous settings are restored on exit, including on exceptions.

    Note:
        Create and use FP8 caches inside an FP8 scope. Using such a cache
        outside an enabled scope raises ``RuntimeError``. Exiting neither
        converts nor clears caches; they can be reused in another FP8 scope.
        FP8 is not guaranteed to improve prediction latency.

    Example:
        >>> with sdm.optimize(attention="fp8"):
        ...     model.fit(context, labels)
        ...     predictions = model.predict(queries)
    """
    setting = (
        _attention_quantization.get() if attention == "inherit" else attention
    )
    token = _attention_quantization.set(setting)
    try:
        yield
    finally:
        _attention_quantization.reset(token)
