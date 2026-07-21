from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from torch import Tensor, nn

from sdm.explain.result import (
    Explanation,
    ExplanationMode,
    InputSite,
    OutputIndex,
)
from sdm.tensor.table import TableTensor


class UnsupportedExplanationError(RuntimeError):
    r"""Raised when a model cannot satisfy a method's requirements."""


@dataclass(frozen=True)
class ExplanationRequirements:
    r"""Model execution behavior required by an explanation method.

    Args:
        gradients: Whether model evaluation needs to preserve autograd.
        supported_modes: Model execution paths implemented by the method.
        input_space: Feature space presented to the method.
    """

    gradients: bool = False
    supported_modes: frozenset[ExplanationMode] = frozenset(
        {ExplanationMode.full_context}
    )
    input_space: Literal["raw", "processed"] = "processed"

    def __post_init__(self) -> None:
        if not isinstance(self.gradients, bool):
            raise TypeError("'gradients' needs to be a boolean")
        if not isinstance(self.supported_modes, frozenset) or not all(
            isinstance(mode, ExplanationMode) for mode in self.supported_modes
        ):
            raise TypeError(
                "'supported_modes' needs to be a frozenset of "
                "'ExplanationMode' values"
            )
        if len(self.supported_modes) == 0:
            raise ValueError("'supported_modes' needs to be non-empty")
        if self.input_space not in ("raw", "processed"):
            raise ValueError("'input_space' needs to be 'raw' or 'processed'")


ExplanationInputs: TypeAlias = Mapping[InputSite, TableTensor]
ExplanationReplacements: TypeAlias = Mapping[InputSite, Tensor]
ExplanationCallable: TypeAlias = Callable[
    [ExplanationReplacements | None],
    TableTensor,
]


class ExplanationMethod(ABC):
    r"""Base class for independently extensible explanation algorithms."""

    requirements: ExplanationRequirements = ExplanationRequirements()

    @property
    def name(self) -> str:
        r"""Stable method name recorded in explanation results."""
        return self.__class__.__name__

    def validate_model(
        self,
        model: nn.Module,
        *,
        mode: ExplanationMode,
    ) -> None:
        r"""Validate method-specific model compatibility.

        Generic execution requirements are validated by the model entry point.
        Methods only need to override this for specialized model interfaces.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("'model' needs to be a torch module")
        if not isinstance(mode, ExplanationMode):
            raise TypeError("'mode' needs to be an 'ExplanationMode'")

    @abstractmethod
    def explain(
        self,
        *,
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        r"""Explain one selected prediction using a prepared evaluation."""
        raise NotImplementedError
