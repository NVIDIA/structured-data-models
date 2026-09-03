# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import sdm.processing as sp
from sdm.processing import EnsembleProcessor, Processor


@dataclass(init=False, repr=False)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe bundles three processing pipelines, one per role the data plays
    relative to the model:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels transformed forward before the model. Regression
      predictions are inverted through this pipeline; classification outputs
      are reconstructed from the fitted target categories instead.
    - ``output``: transforms member outputs after they have been mapped to a
      common class or target space and stacked as ``[E, ..., R, O]``. An
      explicit dimension-changing step such as
      :class:`~sdm.processing.ReduceEstimators` removes ``E``; without one,
      the output remains stacked. Steps before the reducer must support
      stacked outputs, while steps after it receive already-reduced outputs.

    Each pipeline exposes ``fit``/``transform``/``fit_transform`` and, when its
    steps are invertible, ``inverse_transform``. Call them directly, e.g.
    ``recipe.features.transform(table)`` or
    ``recipe.target.inverse_transform(prediction)``. Recipes do not infer each
    step's non-finite input contract; order steps so values are imputed before
    processors that do not explicitly document non-finite support.

    Args:
        features: Steps applied to model inputs before the model.
        target: Steps applied to labels. Invertible numerical target steps map
            regression output back to the original space.
        output: Steps applied to stacked member outputs after member-local
            mappings. Estimator reduction, when desired, is an explicit step
            in this pipeline.
    """

    features: EnsembleProcessor
    target: EnsembleProcessor
    output: EnsembleProcessor

    def __init__(
        self,
        features: Processor | Iterable[Processor] | None = None,
        target: Processor | Iterable[Processor] | None = None,
        output: Processor | Iterable[Processor] | None = None,
    ) -> None:

        self.features = EnsembleProcessor.as_processor(
            sp.Identity() if features is None else features
        )
        self.target = EnsembleProcessor.as_processor(
            sp.Identity() if target is None else target
        )
        self.output = EnsembleProcessor.as_processor(
            sp.Identity() if output is None else output
        )

        self._validate_target()
        self._validate_output()

    def prepend_features(self, processor: object) -> Self:
        """Prepend a processor to the feature pipeline."""
        self.features = processor + self.features
        return self

    def append_features(self, processor: object) -> Self:
        """Append a processor to the feature pipeline."""
        self.features = self.features + processor
        return self

    def prepend_target(self, processor: object) -> Self:
        """Prepend a processor to the target pipeline."""
        self.target = processor + self.target
        self._validate_target()
        return self

    def append_target(self, processor: object) -> Self:
        """Append a processor to the target pipeline."""
        self.target = self.target + processor
        self._validate_target()
        return self

    def prepend_output(self, processor: object) -> Self:
        """Prepend a processor to the output pipeline."""
        self.output = processor + self.output
        self._validate_output()
        return self

    def append_output(self, processor: object) -> Self:
        """Append a processor to the output pipeline."""
        self.output = self.output + processor
        self._validate_output()
        return self

    def _validate_target(self) -> None:
        if any(isinstance(m, sp.TaskDispatch) for m in self.target.modules()):
            raise ValueError(
                "'TaskDispatch' is not supported in 'Recipe.target'"
            )
        if any(isinstance(m, sp.TableDispatch) for m in self.target.modules()):
            raise ValueError(
                "'TableDispatch' is not supported in 'Recipe.target'"
            )

    def _validate_output(self) -> None:
        if any(isinstance(m, sp.TableDispatch) for m in self.output.modules()):
            raise ValueError(
                "'TableDispatch' is not supported in 'Recipe.output'"
            )
        if self.output.requires_fit:
            raise ValueError("'Recipe.output' should not require fitting")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  features={self.features.__repr__(indent=2)[2:]},\n"
            f"  target={self.target.__repr__(indent=2)[2:]},\n"
            f"  output={self.output.__repr__(indent=2)[2:]},\n"
            ")"
        )
