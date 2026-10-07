# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importable small test models for spawned process correctness checks."""

from typing import Any, cast

import torch

import sdm.processing as sp
from sdm import Recipe, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel


class _RandomCacheModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_multi_target = False
    supports_related_tables = False

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, device=device))
        self.fail = False
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe(features=sp.ShuffleColumns(), target=sp.Standardize())

    def _forward(
        self,
        *,
        cache: Cache,
        x_context: TableTensor | None,
        x_query: TableTensor | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        if self.fail:
            raise ValueError("injected model failure")
        if cache.is_recording:
            assert x_context is not None
            cache["bias"] = torch.rand(
                1, device=x_context.device, generator=generator
            )
            return x_context
        assert x_query is not None
        return TableTensor(
            numerical=x_query.numerical[..., :1] * self.weight
            + cast(torch.Tensor, cache["bias"])
        )
