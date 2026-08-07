from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sdm.processing import (
    EnsembleProcessor,
    Identity,
    Processor,
    TaskDispatch,
)


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
    processors that do not explicitly document non-finite support. A
    :class:`~sdm.processing.TaskDispatch` in ``features`` or ``output`` is
    selected from the transformed target during model execution.

    Bind a recipe to context data with :meth:`bind` to obtain a reusable
    execution for query, inverse-target, and output transforms.

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
            Identity() if features is None else features
        )
        self.target = EnsembleProcessor.as_processor(
            Identity() if target is None else target
        )
        self.output = EnsembleProcessor.as_processor(
            Identity() if output is None else output
        )

        if any(isinstance(m, TaskDispatch) for m in self.target.modules()):
            raise ValueError(
                "'TaskDispatch' is not supported in 'Recipe.target'"
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
