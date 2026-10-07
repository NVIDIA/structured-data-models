# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from enum import Enum
from typing import Any

import torch
from torch import Tensor


def _tensor_cache_hash(tensor: Tensor) -> str:
    """Hash wrapper metadata without pickling symbolic shape environments."""
    seen: dict[int, int] = {}
    storages: dict[torch.UntypedStorage, int] = {}

    def normalize(value: Any) -> Any:
        if isinstance(value, Tensor):
            # Preserve repeated-leaf aliases without using process addresses.
            if id(value) in seen:
                return ("reference", seen[id(value)])
            seen[id(value)] = len(seen)
            metadata = (
                type(value).__module__,
                type(value).__qualname__,
                normalize(tuple(value.shape)),
                normalize(tuple(value.stride())),
                normalize(value.storage_offset()),
                str(value.dtype),
                str(value.device),
                str(value.layout),
                value.requires_grad,
                value.is_inference(),
                value.is_conj(),
                value.is_neg(),
            )
            if hasattr(value, "__tensor_flatten__"):
                names, context = value.__tensor_flatten__()
                return (
                    "wrapper",
                    metadata,
                    normalize(context),
                    tuple(
                        (name, normalize(getattr(value, name)))
                        for name in names
                    ),
                )
            # Distinct views may alias despite being different Tensor objects.
            # Only traversal-order group numbers enter the persistent hash.
            storage = value.untyped_storage()
            group = storages.setdefault(storage, len(storages))
            return ("tensor", metadata, group)
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return (
                "symbol",
                type(value).__name__,
                str(value.node.expr),
                normalize(value.node.hint),
            )
        if isinstance(value, Enum):
            return (
                "enum",
                type(value).__module__,
                type(value).__qualname__,
                normalize(value.value),
            )
        if isinstance(value, type):
            return ("type", value.__module__, value.__qualname__)
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(map(normalize, value)))
        if isinstance(value, dict):
            return (
                "dict",
                tuple((normalize(k), normalize(v)) for k, v in value.items()),
            )
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported tensor cache metadata: {type(value)}")

    payload = ("sdm-tensor-metadata-v2", normalize(tensor))
    return hashlib.blake2b(repr(payload).encode(), digest_size=32).hexdigest()
