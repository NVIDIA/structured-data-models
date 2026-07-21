from dataclasses import dataclass
from enum import Enum
from typing import Literal

from torch import Tensor

from sdm.stype import Stype
from sdm.tensor.table import TableTensor


class ExplanationMode(str, Enum):
    r"""Model execution path used for an explanation."""

    full_context = "full_context"
    fitted = "fitted"


@dataclass(frozen=True)
class InputSite:
    r"""Location of one table or processed numerical input block.

    Args:
        split: Whether the input belongs to the context or query split.
        table: Related-table name, or ``None`` for the task table.
    """

    split: Literal["context", "query"]
    table: str | None = None

    def __post_init__(self) -> None:
        if self.split not in ("context", "query"):
            raise ValueError("'split' needs to be 'context' or 'query'")
        if self.table is not None:
            if not isinstance(self.table, str):
                raise TypeError("'table' needs to be a string or None")
            if len(self.table) == 0:
                raise ValueError("'table' needs to be non-empty when provided")


@dataclass(frozen=True)
class OutputIndex:
    r"""Select one scalar from a two-dimensional numerical prediction.

    Args:
        row: Positional prediction row.
        column: Numerical output column name or position.
    """

    row: int
    column: int | str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.row, int)
            or isinstance(self.row, bool)
            or self.row < 0
        ):
            raise ValueError("'row' needs to be a non-negative integer")
        if not isinstance(self.column, int | str) or isinstance(
            self.column, bool
        ):
            raise ValueError("'column' needs to be a string or integer")
        if isinstance(self.column, int) and self.column < 0:
            raise ValueError(
                "Integer 'column' needs to be a non-negative integer"
            )
        if isinstance(self.column, str) and len(self.column) == 0:
            raise ValueError("String 'column' needs to be non-empty")

    def resolve(self, prediction: TableTensor) -> "ResolvedTarget":
        r"""Resolve this selector against ``prediction``."""
        if prediction.dim() != 2:
            raise ValueError(
                "Expected an unbatched, two-dimensional prediction "
                f"(got {prediction.dim()} dimensions)"
            )
        if self.row >= prediction.size(-2):
            raise ValueError("'row' is outside the prediction rows")

        columns = prediction.columns.get(Stype.numerical, ())
        if isinstance(self.column, str):
            if self.column not in columns:
                raise ValueError(
                    f"Unknown numerical prediction column '{self.column}'"
                )
            column = columns.index(self.column)
        else:
            column = self.column
            if column >= len(columns):
                raise ValueError(
                    "'column' is outside the numerical prediction columns"
                )

        return ResolvedTarget(
            row=self.row,
            column=column,
            column_name=columns[column],
        )


@dataclass(frozen=True)
class ResolvedTarget:
    r"""Resolved scalar prediction target."""

    row: int
    column: int
    column_name: str
    output_space: Literal["prediction"] = "prediction"

    def __post_init__(self) -> None:
        if self.output_space != "prediction":
            raise ValueError("'output_space' needs to be 'prediction'")

    def select(self, prediction: TableTensor) -> Tensor:
        r"""Select the scalar target from ``prediction``."""
        resolved = OutputIndex(
            row=self.row,
            column=self.column,
        ).resolve(prediction)
        if resolved != self:
            raise ValueError("Resolved target does not match the prediction")
        return prediction.numerical[self.row, self.column]


@dataclass(frozen=True)
class FeatureAttribution:
    r"""Schema-aligned attribution values for one model input site."""

    site: InputSite
    values: TableTensor
    score_kind: str
    input_space: Literal["raw", "processed"]
    signed: bool = True
    normalization: Literal["none", "global_max_abs"] = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.site, InputSite):
            raise TypeError("'site' needs to be an 'InputSite'")
        if not isinstance(self.values, TableTensor):
            raise TypeError("'values' needs to be a 'TableTensor'")
        invalid = self.values.active_stypes - {Stype.numerical, Stype.id}
        if len(invalid) > 0:
            raise ValueError(
                "Feature attribution values may only contain numerical "
                "scores and identifiers"
            )
        if not self.values.numerical.is_floating_point():
            raise TypeError(
                "Feature attribution values need to be floating point"
            )
        if not isinstance(self.score_kind, str):
            raise TypeError("'score_kind' needs to be a string")
        if len(self.score_kind) == 0:
            raise ValueError("'score_kind' needs to be non-empty")
        if self.input_space not in ("raw", "processed"):
            raise ValueError("'input_space' needs to be 'raw' or 'processed'")
        if not isinstance(self.signed, bool):
            raise TypeError("'signed' needs to be a boolean")
        if self.normalization not in ("none", "global_max_abs"):
            raise ValueError(
                "'normalization' needs to be 'none' or 'global_max_abs'"
            )


class ExplanationDiagnostic:
    r"""Base class for method-specific typed diagnostics."""


@dataclass(frozen=True)
class Explanation:
    r"""Prediction and attributions explaining one resolved target."""

    prediction: TableTensor
    target: ResolvedTarget
    method: str
    mode: ExplanationMode
    attributions: tuple[FeatureAttribution, ...] = ()
    diagnostics: tuple[ExplanationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, TableTensor):
            raise TypeError("'prediction' needs to be a 'TableTensor'")
        if not isinstance(self.target, ResolvedTarget):
            raise TypeError("'target' needs to be a 'ResolvedTarget'")
        if not isinstance(self.method, str):
            raise TypeError("'method' needs to be a string")
        if len(self.method) == 0:
            raise ValueError("'method' needs to be non-empty")
        if not isinstance(self.mode, ExplanationMode):
            raise TypeError("'mode' needs to be an 'ExplanationMode'")
        if not isinstance(self.attributions, tuple):
            raise TypeError("'attributions' needs to be a tuple")
        if not all(
            isinstance(attribution, FeatureAttribution)
            for attribution in self.attributions
        ):
            raise TypeError(
                "Every attribution needs to be a 'FeatureAttribution'"
            )
        if not isinstance(self.diagnostics, tuple):
            raise TypeError("'diagnostics' needs to be a tuple")
        if not all(
            isinstance(diagnostic, ExplanationDiagnostic)
            for diagnostic in self.diagnostics
        ):
            raise TypeError(
                "Every diagnostic needs to be an 'ExplanationDiagnostic'"
            )
        self.target.select(self.prediction)
