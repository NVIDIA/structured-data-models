# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        output=sp.ReduceEstimators(method="mean"),
    )


def _is_identity_target(processor: sp.EnsembleProcessor) -> bool:
    if isinstance(processor, sp.EnsembleProcessorAdapter):
        return isinstance(processor.processor, sp.Identity)
    if isinstance(processor, sp.Sequential):
        return all(_is_identity_target(child) for child in processor)
    return False


def _validate_recipe(recipe: sp.Recipe | None) -> None:
    if recipe is None or _is_identity_target(recipe.target):
        return
    raise ValueError(
        "TimesFM3 does not support transformations in 'Recipe.target' "
        "because it returns multiple quantiles per target. Leave the target "
        "pipeline unchanged; feature and output pipelines remain "
        "configurable."
    )
