from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.explain.base import (
    ExplanationCallable,
    ExplanationInputs,
    ExplanationMethod,
    ExplanationMode,
    ExplanationRequirements,
    UnsupportedExplanationError,
)
from sdm.explain.result import (
    Explanation,
    ExplanationDiagnostic,
    FeatureAttribution,
    InputSite,
    OutputIndex,
    ResolvedTarget,
)

_Baseline: TypeAlias = float | int | Tensor | TableTensor


@dataclass(frozen=True)
class IntegratedGradientsDiagnostics(ExplanationDiagnostic):
    r"""Captum completeness information for one explanation.

    Args:
        reference_prediction: Prediction at the configured baselines while
            every unselected input remains fixed.
        convergence_delta: Signed Captum completeness residual.
        varied_sites: Input sites interpolated from baseline to input.
        n_steps: Number of integration steps.
    """

    reference_prediction: TableTensor
    convergence_delta: Tensor
    varied_sites: tuple[InputSite, ...]
    n_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference_prediction, TableTensor):
            raise TypeError(
                "'reference_prediction' needs to be a 'TableTensor'"
            )
        if not isinstance(self.convergence_delta, Tensor):
            raise TypeError("'convergence_delta' needs to be a tensor")
        if self.convergence_delta.numel() != 1:
            raise ValueError("'convergence_delta' needs one value")
        if not isinstance(self.varied_sites, tuple) or not all(
            isinstance(site, InputSite) for site in self.varied_sites
        ):
            raise TypeError(
                "'varied_sites' needs to be a tuple of input sites"
            )
        if len(self.varied_sites) == 0:
            raise ValueError("'varied_sites' needs to be non-empty")
        if len(set(self.varied_sites)) != len(self.varied_sites):
            raise ValueError("'varied_sites' needs to contain unique sites")
        if (
            not isinstance(self.n_steps, int)
            or isinstance(self.n_steps, bool)
            or self.n_steps < 1
        ):
            raise ValueError("'n_steps' needs to be a positive integer")


class CaptumIntegratedGradients(ExplanationMethod):
    r"""Apply Captum Integrated Gradients to processed numerical inputs.

    Args:
        baselines: Optional processed baselines keyed by input site. A
            scalar creates a constant numerical baseline, a ``Tensor``
            supplies numerical values directly, and a ``TableTensor`` supplies
            its numerical block. When omitted, every floating-point input site
            uses zero. When provided, only listed sites vary and unlisted sites
            remain fixed at their input values.
        n_steps: Number of Gauss-Legendre integration steps.
    """

    requirements = ExplanationRequirements(gradients=True)

    def __init__(
        self,
        baselines: Mapping[InputSite, _Baseline] | None = None,
        *,
        n_steps: int = 50,
    ) -> None:
        if baselines is not None:
            if not isinstance(baselines, Mapping):
                raise TypeError("'baselines' needs to be a mapping or None")
            if len(baselines) == 0:
                raise ValueError("'baselines' needs to be non-empty")
            if not all(isinstance(site, InputSite) for site in baselines):
                raise TypeError(
                    "'baselines' keys need to be 'InputSite' values"
                )
            if not all(
                isinstance(value, Tensor | int | float)
                and not isinstance(value, bool)
                for value in baselines.values()
            ):
                raise TypeError(
                    "'baselines' values need to be scalars, tensors, or "
                    "TableTensors"
                )
            baselines = dict(baselines)
        if (
            not isinstance(n_steps, int)
            or isinstance(n_steps, bool)
            or n_steps < 1
        ):
            raise ValueError("'n_steps' needs to be a positive integer")
        self.baselines = baselines
        self.n_steps = n_steps

    def explain(
        self,
        *,
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        r"""Return baseline-relative attributions for one prediction."""
        if mode is not ExplanationMode.full_context:
            raise UnsupportedExplanationError(
                "CaptumIntegratedGradients requires full-context execution"
            )
        integrated_gradients = _load_integrated_gradients()
        resolved = target.resolve(prediction)
        parts = _prepare_parts(inputs, self.baselines)
        values = tuple(part.value for part in parts)
        baselines = tuple(part.baseline for part in parts)
        adapter = _ScalarAdapter(
            evaluate=evaluate,
            parts=parts,
            target=resolved,
        )

        with torch.no_grad():
            endpoint_prediction = _detach_prediction(
                evaluate(_replacements(values, parts))
            )
            reference_prediction = _detach_prediction(
                evaluate(_replacements(baselines, parts))
            )
        if not endpoint_prediction.allclose(
            prediction,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                "Captum endpoint evaluation does not match the canonical "
                "prediction"
            )

        captum = integrated_gradients(adapter)
        attributions, delta = captum.attribute(
            values,
            baselines=baselines,
            n_steps=self.n_steps,
            internal_batch_size=1,
            return_convergence_delta=True,
        )
        attribution_tensors = _attribution_tuple(
            attributions,
            expected=len(parts),
        )
        if not isinstance(delta, Tensor):
            raise TypeError(
                "Captum Integrated Gradients returned an invalid delta"
            )

        return Explanation(
            prediction=_detach_prediction(prediction),
            target=resolved,
            method=self.name,
            mode=mode,
            attributions=tuple(
                FeatureAttribution(
                    site=part.site,
                    values=_score_table(
                        inputs[part.site],
                        attribution.detach().reshape(part.shape),
                    ),
                    score_kind="integrated_gradients",
                    input_space="processed",
                )
                for part, attribution in zip(parts, attribution_tensors)
            ),
            diagnostics=(
                IntegratedGradientsDiagnostics(
                    reference_prediction=reference_prediction,
                    convergence_delta=delta.detach(),
                    varied_sites=tuple(part.site for part in parts),
                    n_steps=self.n_steps,
                ),
            ),
        )


@dataclass(frozen=True)
class _FlatPart:
    site: InputSite
    shape: torch.Size
    value: Tensor
    baseline: Tensor


class _ScalarAdapter:
    def __init__(
        self,
        *,
        evaluate: ExplanationCallable,
        parts: tuple[_FlatPart, ...],
        target: ResolvedTarget,
    ) -> None:
        self.evaluate = evaluate
        self.parts = parts
        self.target = target

    def __call__(self, *values: Tensor) -> Tensor:
        prediction = self.evaluate(_replacements(values, self.parts))
        return self.target.select(prediction).reshape(1)


def _prepare_parts(
    inputs: ExplanationInputs,
    configured: Mapping[InputSite, _Baseline] | None,
) -> tuple[_FlatPart, ...]:
    tables = dict(inputs)
    if any(not isinstance(site, InputSite) for site in tables):
        raise TypeError("Input keys need to be 'InputSite' values")
    if any(not isinstance(table, TableTensor) for table in tables.values()):
        raise TypeError("Captum inputs need TableTensor values")
    if any(table.dim() != 2 for table in tables.values()):
        raise UnsupportedExplanationError(
            "CaptumIntegratedGradients does not support batched inputs"
        )

    if configured is None:
        selected = {
            site: torch.zeros_like(table.numerical)
            for site, table in tables.items()
            if table.numerical.is_floating_point()
        }
    else:
        unknown = configured.keys() - tables.keys()
        if len(unknown) > 0:
            raise UnsupportedExplanationError(
                f"Unknown Captum baseline input site: {next(iter(unknown))!r}"
            )
        selected = {
            site: _baseline_tensor(value, tables[site], site=site)
            for site, value in configured.items()
        }

    parts: list[_FlatPart] = []
    for site, baseline in selected.items():
        table = tables[site]
        value = table.numerical
        if not value.is_floating_point():
            raise UnsupportedExplanationError(
                f"Captum input at {site!r} needs floating-point numerical "
                "values"
            )
        _validate_baseline(baseline, value, site=site)
        parts.append(
            _FlatPart(
                site=site,
                shape=value.shape,
                value=value.detach().clone().reshape(1, -1),
                baseline=baseline.detach().clone().reshape(1, -1),
            )
        )
    if len(parts) == 0:
        raise UnsupportedExplanationError(
            "CaptumIntegratedGradients found no floating-point numerical "
            "input sites"
        )
    return tuple(parts)


def _baseline_tensor(
    baseline: _Baseline,
    table: TableTensor,
    *,
    site: InputSite,
) -> Tensor:
    if isinstance(baseline, int | float) and not isinstance(baseline, bool):
        return torch.full_like(table.numerical, baseline)
    if isinstance(baseline, TableTensor):
        if baseline.columns[Stype.numerical] != table.columns[Stype.numerical]:
            raise ValueError(
                f"Captum baseline at {site!r} needs the same numerical "
                "columns as the processed input"
            )
        if baseline.columns[Stype.id] != table.columns[
            Stype.id
        ] or not baseline.id.equal(table.id):
            raise ValueError(
                f"Captum baseline at {site!r} needs the same identifiers "
                "as the processed input"
            )
        return baseline.numerical
    if not isinstance(baseline, Tensor):
        raise TypeError(
            f"Captum baseline at {site!r} needs to be a scalar, tensor, or "
            "TableTensor"
        )
    return baseline


def _validate_baseline(
    baseline: Tensor,
    value: Tensor,
    *,
    site: InputSite,
) -> None:
    if baseline.shape != value.shape:
        raise ValueError(
            f"Captum baseline at {site!r} needs shape {tuple(value.shape)} "
            f"(got {tuple(baseline.shape)})"
        )
    if baseline.device != value.device:
        raise ValueError(
            f"Captum baseline at {site!r} needs device '{value.device}' "
            f"(got '{baseline.device}')"
        )
    if baseline.dtype != value.dtype:
        raise ValueError(
            f"Captum baseline at {site!r} needs dtype '{value.dtype}' "
            f"(got '{baseline.dtype}')"
        )


def _replacements(
    values: tuple[Tensor, ...],
    parts: tuple[_FlatPart, ...],
) -> Mapping[InputSite, Tensor]:
    if len(values) != len(parts):
        raise UnsupportedExplanationError(
            "Captum adapter requires one value per varied input site"
        )
    replacements: dict[InputSite, Tensor] = {}
    for value, part in zip(values, parts):
        if value.dim() != 2 or value.shape != (1, part.shape.numel()):
            raise UnsupportedExplanationError(
                "Captum evaluation requires one interpolation point for "
                f"{part.site!r}"
            )
        replacements[part.site] = value[0].reshape(part.shape)
    return replacements


def _attribution_tuple(
    attributions: Any,
    *,
    expected: int,
) -> tuple[Tensor, ...]:
    if isinstance(attributions, Tensor):
        values = (attributions,)
    elif isinstance(attributions, tuple) and all(
        isinstance(value, Tensor) for value in attributions
    ):
        values = cast(tuple[Tensor, ...], attributions)
    else:
        raise TypeError(
            "Captum Integrated Gradients returned invalid attributions"
        )
    if len(values) != expected:
        raise TypeError(
            "Captum Integrated Gradients returned an unexpected number of "
            "attributions"
        )
    return values


def _score_table(table: TableTensor, scores: Tensor) -> TableTensor:
    return table.select_stypes((Stype.numerical, Stype.id)).replace_blocks(
        numerical=scores
    )


def _detach_prediction(prediction: TableTensor) -> TableTensor:
    return prediction.replace_blocks(numerical=prediction.numerical.detach())


def _load_integrated_gradients() -> type[Any]:
    try:
        from captum.attr import IntegratedGradients
    except ModuleNotFoundError as error:
        if error.name != "captum":
            raise
        raise ImportError(
            "CaptumIntegratedGradients requires the optional Captum "
            "dependency. Install 'structured-data-models[captum]'."
        ) from error
    return IntegratedGradients
