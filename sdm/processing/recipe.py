from collections.abc import Iterable
from dataclasses import dataclass, field

from sdm.processing.base import Processor
from sdm.processing.pipeline import Pipeline
from sdm.tensor import TableTensor


@dataclass(frozen=True)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe bundles three :class:`~sdm.processing.Pipeline` objects, one per
    role the data plays relative to the model:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels, transformed forward before the model and inverted
      after it (predictions back to the original space).
    - ``output``: shape-preserving cleanup of the model output.

    Each pipeline exposes ``fit``/``transform``/``fit_transform`` and, when its
    steps are invertible, ``inverse_transform``. Call them directly, e.g.
    ``recipe.features.transform(table)`` or
    ``recipe.target.inverse_transform(prediction)``.

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
            self.features.fit_transform(features),
            self.target.fit_transform(target),
        )

    def __repr__(self) -> str:
        roles = "\n".join(
            f"  {name}: "
            + (
                " -> ".join(step.__class__.__name__ for step in pipeline)
                or "identity"
            )
            for name, pipeline in (
                ("features", self.features),
                ("target", self.target),
                ("output", self.output),
            )
        )
        return f"Recipe(\n{roles}\n)"
