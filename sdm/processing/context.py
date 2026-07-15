from dataclasses import dataclass
from typing import Literal, Protocol

from sdm.tensor import TableTensor


class TargetInverse(Protocol):
    """Target inverse required while decoding regression output."""

    def inverse_transform(self, table: TableTensor) -> TableTensor:
        """Map transformed model output to the original target space."""
        ...


@dataclass(frozen=True)
class RecipeContext:
    """Fitted state for one estimator while processing ensemble output.

    A model creates one context after fitting each estimator-local Recipe.
    Output Processors receive the contexts in estimator order together with
    the stacked raw model outputs.

    Args:
        estimator_index: Position of the estimator in the stacked output.
        task: Task selected from the transformed target.
        class_indices: Model-output columns that restore the original class
            order for classification.
        target_inverse: Fitted target inverse for regression.
    """

    estimator_index: int
    task: Literal["classification", "regression"]
    class_indices: tuple[int, ...] | None = None
    target_inverse: TargetInverse | None = None
