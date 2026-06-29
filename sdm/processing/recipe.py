from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from torch import Tensor
from typing_extensions import Self

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class Pipeline:
    """Ordered sequence of processing stages for one recipe slot.

    Args:
        stages: Ordered processing stages. ``None`` creates an empty identity
            pipeline.
    """

    def __init__(
        self,
        stages: Iterable[Processor] | None = None,
    ) -> None:
        self.stages = tuple(_validate_stage(stage) for stage in stages or ())

    def fit(self, input: TableTensor) -> Self:
        """Fit stages in order using the numerical block of ``input``."""
        self._fit(input, slot="pipeline")
        return self

    def _fit(
        self,
        input: TableTensor,
        *,
        slot: str,
    ) -> Self:
        numerical = input.numerical
        for position, stage in enumerate(self.stages):
            try:
                stage.fit(numerical)
                numerical = _validate_output(stage.transform(numerical))
            except Exception as exc:
                raise _stage_error(exc, slot, position, stage) from exc
        return self

    def transform(self, input: TableTensor) -> TableTensor:
        """Transform ``input`` by applying stages to its numerical block."""
        return self._transform(input, slot="pipeline")

    def _transform(self, input: TableTensor, *, slot: str) -> TableTensor:
        if len(self.stages) == 0:
            return input
        numerical = input.numerical
        for position, stage in enumerate(self.stages):
            try:
                numerical = _validate_output(stage.transform(numerical))
            except Exception as exc:
                raise _stage_error(exc, slot, position, stage) from exc
        return _with_stage_numerical(input, numerical, slot, position, stage)

    def fit_transform(self, input: TableTensor) -> TableTensor:
        """Fit and transform ``input`` by threading stages in order."""
        return self._fit_transform(input, slot="pipeline")

    def _fit_transform(self, input: TableTensor, *, slot: str) -> TableTensor:
        if len(self.stages) == 0:
            return input
        numerical = input.numerical
        for position, stage in enumerate(self.stages):
            try:
                numerical = _validate_output(stage.fit_transform(numerical))
            except Exception as exc:
                raise _stage_error(exc, slot, position, stage) from exc
        return _with_stage_numerical(input, numerical, slot, position, stage)

    def inverse_transform(self, input: TableTensor) -> TableTensor:
        """Apply invertible stages in reverse order to ``input``."""
        return self._inverse_transform(input, slot="pipeline")

    def _inverse_transform(
        self,
        input: TableTensor,
        *,
        slot: str,
    ) -> TableTensor:
        if len(self.stages) == 0:
            return input
        numerical = input.numerical
        for position, stage in reversed(tuple(enumerate(self.stages))):
            if not isinstance(stage, InvertibleMixin):
                raise _stage_error(
                    TypeError(
                        "Expected invertible stage for inverse_transform"
                    ),
                    slot,
                    position,
                    stage,
                )
            try:
                numerical = _validate_output(
                    stage.inverse_transform(numerical)
                )
            except Exception as exc:
                raise _stage_error(exc, slot, position, stage) from exc
        return _with_stage_numerical(input, numerical, slot, position, stage)

    def describe(self) -> str:
        """Return a human-readable stage-order summary."""
        if len(self.stages) == 0:
            return "<empty>"
        return " -> ".join(stage.__class__.__name__ for stage in self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self) -> Iterator[Processor]:
        return iter(self.stages)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.describe()})"


@dataclass(frozen=True)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe owns deterministic preprocessing composition only. It does not own
    sampling, augmentation, model calls, or experiment control flow.

    Args:
        preprocess: Stages applied before the model.
        target: Target-side stages; inverse conversion runs after the model.
        postprocess: Stages applied to model output after target inverse.
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
        """Fit the preprocess slot on ``input`` and return this recipe."""
        self.preprocess._fit(input, slot="preprocess")
        return self

    def transform_preprocess(self, input: TableTensor) -> TableTensor:
        """Run feature preprocessing before an external model call."""
        return self.preprocess._transform(input, slot="preprocess")

    def fit_transform_preprocess(self, input: TableTensor) -> TableTensor:
        """Fit and run feature preprocessing before an external model call."""
        return self.preprocess._fit_transform(input, slot="preprocess")

    def inverse_transform_target(self, input: TableTensor) -> TableTensor:
        """Run target inverse conversion after an external model call."""
        return self.target._inverse_transform(input, slot="target")

    def transform_postprocess(self, input: TableTensor) -> TableTensor:
        """Run shape-preserving postprocessing after an external model call."""
        return self.postprocess._transform(input, slot="postprocess")

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


def _validate_stage(stage: Processor) -> Processor:
    if not isinstance(stage, Processor):
        raise TypeError(
            "Expected a Processor stage "
            f"(got '{stage.__class__.__name__}')"
        )
    return stage


def _validate_output(output: object) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError(
            "Expected numerical block transform to return a Tensor "
            f"(got '{type(output).__name__}')"
        )
    return output


def _with_stage_numerical(
    input: TableTensor,
    numerical: Tensor,
    slot: str,
    position: int,
    stage: Processor,
) -> TableTensor:
    try:
        return _with_numerical(input, numerical)
    except Exception as exc:
        raise _stage_error(exc, slot, position, stage) from exc


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


def _stage_error(
    exc: Exception,
    slot: str,
    position: int,
    stage: Processor,
) -> Exception:
    message = (
        f"[slot={slot}, stage={position}, "
        f"name={stage.__class__.__name__}] {exc}"
    )
    try:
        return exc.__class__(message)
    except Exception:
        return RuntimeError(message)
