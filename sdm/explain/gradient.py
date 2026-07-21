from collections.abc import Mapping
from typing import Literal

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
    FeatureAttribution,
    InputSite,
    OutputIndex,
)


class GradientSensitivity(ExplanationMethod):
    r"""Differentiate one prediction with respect to processed inputs.

    Args:
        magnitude: Return absolute gradient magnitudes instead of signed
            gradients.
        normalization: ``"global_max_abs"`` divides every emitted score by
            the maximum absolute score across all input sites. All-zero
            scores remain zero.
    """

    requirements = ExplanationRequirements(gradients=True)

    def __init__(
        self,
        *,
        magnitude: bool = False,
        normalization: Literal["none", "global_max_abs"] = "none",
    ) -> None:
        if not isinstance(magnitude, bool):
            raise TypeError("'magnitude' needs to be a boolean")
        if normalization not in ("none", "global_max_abs"):
            raise ValueError(
                "'normalization' needs to be 'none' or 'global_max_abs'"
            )
        self.magnitude = magnitude
        self.normalization: Literal["none", "global_max_abs"] = normalization

    def explain(
        self,
        *,
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        r"""Return gradients for one selected final prediction."""
        if mode is not ExplanationMode.full_context:
            raise UnsupportedExplanationError(
                "GradientSensitivity requires full-context execution"
            )
        tables = _input_tables(inputs)
        leaves = {
            site: table.numerical.detach().clone().requires_grad_(True)
            for site, table in tables.items()
            if table.numerical.is_floating_point()
        }
        if len(leaves) == 0:
            raise UnsupportedExplanationError(
                "GradientSensitivity found no floating-point numerical "
                "input sites"
            )

        differentiable_prediction = evaluate(leaves)
        if not differentiable_prediction.allclose(
            prediction,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                "Gradient evaluation does not match the canonical prediction"
            )
        resolved = target.resolve(prediction)
        selected = resolved.select(differentiable_prediction)
        if not selected.requires_grad:
            raise UnsupportedExplanationError(
                "The selected prediction is not differentiable"
            )

        gradients = torch.autograd.grad(
            selected,
            tuple(leaves.values()),
            allow_unused=True,
        )
        if all(gradient is None for gradient in gradients):
            raise UnsupportedExplanationError(
                "The selected prediction does not use any available "
                "numerical input site"
            )
        scores = {
            site: (
                torch.zeros_like(leaves[site])
                if gradient is None
                else gradient.detach()
            )
            for site, gradient in zip(leaves, gradients)
        }
        if self.magnitude:
            scores = {site: score.abs() for site, score in scores.items()}
        if self.normalization == "global_max_abs":
            scores = _normalize_global_max_abs(scores)

        prediction = prediction.replace_blocks(
            numerical=prediction.numerical.detach()
        )
        return Explanation(
            prediction=prediction,
            target=resolved,
            method=self.name,
            mode=mode,
            attributions=tuple(
                FeatureAttribution(
                    site=site,
                    values=_score_table(tables[site], scores[site]),
                    score_kind="gradient",
                    input_space="processed",
                    signed=not self.magnitude,
                    normalization=self.normalization,
                )
                for site in scores
            ),
        )


def _input_tables(
    inputs: ExplanationInputs,
) -> dict[InputSite, TableTensor]:
    tables: dict[InputSite, TableTensor] = {}
    for site, table in inputs.items():
        if not isinstance(site, InputSite):
            raise TypeError("Input keys need to be 'InputSite' values")
        if not isinstance(table, TableTensor):
            raise TypeError(
                "GradientSensitivity inputs need TableTensor values"
            )
        if table.dim() != 2:
            raise UnsupportedExplanationError(
                "GradientSensitivity does not support batched model inputs"
            )
        tables[site] = table
    return tables


def _normalize_global_max_abs(
    scores: Mapping[InputSite, Tensor],
) -> dict[InputSite, Tensor]:
    maxima = [score.abs().amax() for score in scores.values() if score.numel()]
    if len(maxima) == 0:
        return dict(scores)
    maximum = torch.stack(maxima).amax()
    denominator = torch.where(maximum > 0, maximum, maximum.new_ones(()))
    return {site: score / denominator for site, score in scores.items()}


def _score_table(table: TableTensor, scores: Tensor) -> TableTensor:
    return table.select_stypes((Stype.numerical, Stype.id)).replace_blocks(
        numerical=scores
    )
