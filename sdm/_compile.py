# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import torch
from torch import Tensor

from sdm.cache import KVCacheEntry

_compile_body: ContextVar[Callable[..., Any] | None] = ContextVar(
    "_compile_body", default=None
)


def chunk_body(body: Callable[..., Any]) -> Callable[..., Any]:
    r"""Return the function that a chunk loop calls for each chunk.

    This is ``body`` itself, or its compiled version while a function returned
    by :func:`compile_chunk_bodies` runs.
    """
    if torch.compiler.is_dynamo_compiling():
        return body
    compile_body = _compile_body.get()
    return body if compile_body is None else compile_body(body)


def compile_chunk_bodies(
    fn: Callable[..., Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    r"""Compile every :func:`chunk_body` that ``fn`` runs.

    ``fn`` itself stays in Python, so its chunk loops run any number of chunks
    and no graph records their count.

    Args:
        fn: The function whose chunk bodies to compile.
        kwargs: Keyword arguments passed to :func:`torch.compile`.

    Returns:
        A function that is called like ``fn``.
    """
    bodies: dict[Callable[..., Any], Callable[..., Any]] = {}

    def compile_body(body: Callable[..., Any]) -> Callable[..., Any]:
        if body not in bodies:
            bodies[body] = _dynamic_shapes(torch.compile(body, **kwargs))
        return bodies[body]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _compile_body.set(compile_body)
        try:
            return fn(*args, **kwargs)
        finally:
            _compile_body.reset(token)

    return wrapper


def _dynamic_shapes(body: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, out: Tensor | None = None, **kwargs: Any) -> Any:
        args = tuple(_mark_dynamic(arg) for arg in args)
        kwargs = {key: _mark_dynamic(arg) for key, arg in kwargs.items()}
        if out is None:
            return body(*args, **kwargs)
        # PyTorch 2.14.1 cannot compile a write to an input that shares its
        # memory with another input of dynamic size, so the graph writes to a
        # temporary.
        tmp = _mark_dynamic(torch.empty_like(out))
        result = body(*args, out=tmp, **kwargs)
        out.copy_(tmp)
        return (out, *result[1:]) if isinstance(result, tuple) else out

    return wrapper


def _mark_dynamic(arg: Any) -> Any:
    if isinstance(arg, KVCacheEntry):
        return KVCacheEntry(*map(_mark_dynamic, arg))
    if isinstance(arg, Tensor):
        # A new view, so that no graph depends on which arguments are the same
        # tensor. A chunk varies in every dimension its body does not fix.
        arg = arg.view_as(arg)
        torch._dynamo.maybe_mark_dynamic(arg, list(range(arg.dim())))
    return arg
