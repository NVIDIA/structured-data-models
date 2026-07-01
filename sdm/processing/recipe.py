from collections.abc import Iterable
from dataclasses import dataclass, field

from typing_extensions import Self

from sdm.processing.base import Processor
from sdm.processing.pipeline import Pipeline
from sdm.tensor import TableTensor


@dataclass(frozen=True)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe owns deterministic, two-sided transforms keyed by the data each
    phase operates on:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels, transformed forward before the model and inverted
      after it (predictions back to the original space).
    - ``output``: shape-preserving cleanup of the model output.

    Args:
        features: Steps applied to model inputs before the model.
        target: Steps applied to labels; transformed forward before the model
            and inverted after it.
        output: Steps applied to model output after the target inverse.
    """

    features: Pipeline = field(default_factory=Pipeline)
    target: Pipeline = field(default_factory=Pipeline)
    output: Pipeline = field(default_factory=Pipeline)

    def __init__(
        self,
        features: Iterable[Processor] | None = None,
        target: Iterable[Processor] | None = None,
        output: Iterable[Processor] | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "features",
            features if isinstance(features, Pipeline) else Pipeline(features),
        )
        object.__setattr__(
            self,
            "target",
            target if isinstance(target, Pipeline) else Pipeline(target),
        )
        object.__setattr__(
            self,
            "output",
            output if isinstance(output, Pipeline) else Pipeline(output),
        )

    def fit_features(self, table: TableTensor) -> Self:
        """Fit the feature phase on ``table`` and return this recipe."""
        self.features._fit(table, phase="features")
        return self

    def transform_features(self, table: TableTensor) -> TableTensor:
        """Transform model inputs before an external model call."""
        return self.features._transform(table, phase="features")

    def fit_transform_features(self, table: TableTensor) -> TableTensor:
        """Fit and transform model inputs before an external model call."""
        return self.features._fit_transform(table, phase="features")

    def fit_target(self, table: TableTensor) -> Self:
        """Fit the target phase on ``table`` and return this recipe."""
        self.target._fit(table, phase="target")
        return self

    def transform_target(self, table: TableTensor) -> TableTensor:
        """Transform labels into model space before an external model call."""
        return self.target._transform(table, phase="target")

    def fit_transform_target(self, table: TableTensor) -> TableTensor:
        """Fit and transform labels into model space before the model."""
        return self.target._fit_transform(table, phase="target")

    def inverse_transform_target(self, table: TableTensor) -> TableTensor:
        """Run target inverse conversion after an external model call."""
        return self.target._inverse_transform(table, phase="target")

    def transform_output(self, table: TableTensor) -> TableTensor:
        """Run shape-preserving cleanup after an external model call."""
        return self.output._transform(table, phase="output")

    def fit_transform(
        self,
        features: TableTensor,
        target: TableTensor,
    ) -> tuple[TableTensor, TableTensor]:
        """Fit and transform training ``features`` and ``target`` at once.

        Convenience for the common training step. Returns the transformed
        ``(features, target)`` tables, ready to feed the model and its loss.
        """
        return (
            self.fit_transform_features(features),
            self.fit_transform_target(target),
        )

    def __repr__(self) -> str:
        phases = "\n".join(
            f"  {name}: "
            + (
                " -> ".join(step.__class__.__name__ for step in phase)
                or "identity"
            )
            for name, phase in (
                ("features", self.features),
                ("target", self.target),
                ("output", self.output),
            )
        )
        return f"Recipe(\n{phases}\n)"
