from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from torch import Tensor
from typing_extensions import Self

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class Pipeline:
    """Ordered sequence of processing steps for one recipe phase.

    Args:
        steps: Ordered processing steps. ``None`` creates an empty identity
            pipeline.
    """

    def __init__(
        self,
        steps: Iterable[Processor] | None = None,
    ) -> None:
        self.steps = tuple(_validate_step(step) for step in steps or ())

    def fit(self, input: TableTensor) -> Self:
        """Fit steps in order using the numerical block of ``input``."""
        self._fit(input, phase="pipeline")
        return self

    def _fit(
        self,
        input: TableTensor,
        *,
        phase: str,
    ) -> Self:
        numerical = input.numerical
        for position, step in enumerate(self.steps):
            try:
                step.fit(numerical)
                numerical = _validate_output(step.transform(numerical))
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return self

    def transform(self, input: TableTensor) -> TableTensor:
        """Transform ``input`` by applying steps to its numerical block."""
        return self._transform(input, phase="pipeline")

    def _transform(self, input: TableTensor, *, phase: str) -> TableTensor:
        if len(self.steps) == 0:
            return input
        numerical = input.numerical
        for position, step in enumerate(self.steps):
            try:
                numerical = _validate_output(step.transform(numerical))
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_step_numerical(input, numerical, phase, position, step)

    def fit_transform(self, input: TableTensor) -> TableTensor:
        """Fit and transform ``input`` by threading steps in order."""
        return self._fit_transform(input, phase="pipeline")

    def _fit_transform(self, input: TableTensor, *, phase: str) -> TableTensor:
        if len(self.steps) == 0:
            return input
        numerical = input.numerical
        for position, step in enumerate(self.steps):
            try:
                numerical = _validate_output(step.fit_transform(numerical))
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_step_numerical(input, numerical, phase, position, step)

    def inverse_transform(self, input: TableTensor) -> TableTensor:
        """Apply invertible steps in reverse order to ``input``."""
        return self._inverse_transform(input, phase="pipeline")

    def _inverse_transform(
        self,
        input: TableTensor,
        *,
        phase: str,
    ) -> TableTensor:
        if len(self.steps) == 0:
            return input
        numerical = input.numerical
        for position, step in reversed(tuple(enumerate(self.steps))):
            if not isinstance(step, InvertibleMixin):
                raise _step_error(
                    TypeError(
                        "Expected invertible step for inverse_transform"
                    ),
                    phase,
                    position,
                    step,
                )
            try:
                numerical = _validate_output(
                    step.inverse_transform(numerical)
                )
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_step_numerical(input, numerical, phase, position, step)

    def describe(self) -> str:
        """Return a human-readable step-order summary."""
        if len(self.steps) == 0:
            return "identity"
        return " -> ".join(step.__class__.__name__ for step in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Processor]:
        return iter(self.steps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.describe()})"


@dataclass(frozen=True)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe owns deterministic preprocessing composition only. It does not own
    sampling, augmentation, model calls, or experiment control flow.

    Args:
        preprocess: Steps applied before the model.
        target: Target-side steps; inverse conversion runs after the model.
        postprocess: Steps applied to model output after target inverse.
    """

    preprocess: Iterable[Processor] | None = field(default_factory=Pipeline)
    target: Iterable[Processor] | None = field(default_factory=Pipeline)
    postprocess: Iterable[Processor] | None = field(default_factory=Pipeline)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preprocess",
            _coerce_pipeline(self.preprocess),
        )
        object.__setattr__(self, "target", _coerce_pipeline(self.target))
        object.__setattr__(
            self,
            "postprocess",
            _coerce_pipeline(self.postprocess),
        )

    def fit_preprocess(self, input: TableTensor) -> Self:
        """Fit the preprocess phase on ``input`` and return this recipe."""
        self.preprocess._fit(input, phase="preprocess")
        return self

    def transform_preprocess(self, input: TableTensor) -> TableTensor:
        """Run feature preprocessing before an external model call."""
        return self.preprocess._transform(input, phase="preprocess")

    def fit_transform_preprocess(self, input: TableTensor) -> TableTensor:
        """Fit and run feature preprocessing before an external model call."""
        return self.preprocess._fit_transform(input, phase="preprocess")

    def inverse_transform_target(self, input: TableTensor) -> TableTensor:
        """Run target inverse conversion after an external model call."""
        return self.target._inverse_transform(input, phase="target")

    def transform_postprocess(self, input: TableTensor) -> TableTensor:
        """Run shape-preserving postprocessing after an external model call."""
        return self.postprocess._transform(input, phase="postprocess")

    def describe(self) -> str:
        """Return a human-readable summary without running inference."""
        return (
            "Recipe(\n"
            f"  preprocess: {self.preprocess.describe()}\n"
            f"  target: {self.target.describe()}\n"
            f"  postprocess: {self.postprocess.describe()}\n"
            ")"
        )

    def __repr__(self) -> str:
        return self.describe()


def _coerce_pipeline(
    value: Iterable[Processor] | None,
) -> Pipeline:
    if isinstance(value, Pipeline):
        return value
    return Pipeline(value)


def _validate_step(step: object) -> Processor:
    if not isinstance(step, Processor):
        raise TypeError(
            "Expected a Processor step "
            f"(got '{step.__class__.__name__}')"
        )
    return step


def _validate_output(output: object) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError(
            "Expected numerical block transform to return a Tensor "
            f"(got '{type(output).__name__}')"
        )
    return output


def _with_step_numerical(
    input: TableTensor,
    numerical: Tensor,
    phase: str,
    position: int,
    step: Processor,
) -> TableTensor:
    try:
        return _with_numerical(input, numerical)
    except Exception as exc:
        raise _step_error(exc, phase, position, step) from exc


def _with_numerical(input: TableTensor, numerical: Tensor) -> TableTensor:
    if numerical is input.numerical:
        return input
    return input.__class__(
        columns={
            Stype.numerical: input.columns[Stype.numerical],
            Stype.categorical: input.columns[Stype.categorical],
        },
        numerical=numerical,
        categorical=input.categorical,
    )


def _step_error(
    exc: Exception,
    phase: str,
    position: int,
    step: Processor,
) -> Exception:
    message = (
        f"{phase} step {position} "
        f"({step.__class__.__name__}): {exc}"
    )
    try:
        return exc.__class__(message)
    except Exception:
        return RuntimeError(message)
