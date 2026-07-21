"""Experimental explainability interfaces."""

from sdm.explain.result import (
    InputSite,
    ExplanationMode,
    OutputIndex,
    ResolvedTarget,
    FeatureAttribution,
    ExplanationDiagnostic,
    Explanation,
)
from sdm.explain.base import (
    UnsupportedExplanationError,
    ExplanationRequirements,
    ExplanationInputs,
    ExplanationReplacements,
    ExplanationCallable,
    ExplanationMethod,
)
from sdm.explain.gradient import GradientSensitivity
from sdm.explain.captum import (
    IntegratedGradientsDiagnostics,
    CaptumIntegratedGradients,
)

__all__ = [
    "InputSite",
    "OutputIndex",
    "ResolvedTarget",
    "FeatureAttribution",
    "ExplanationDiagnostic",
    "Explanation",
    "UnsupportedExplanationError",
    "ExplanationMode",
    "ExplanationRequirements",
    "ExplanationInputs",
    "ExplanationReplacements",
    "ExplanationCallable",
    "ExplanationMethod",
    "GradientSensitivity",
    "IntegratedGradientsDiagnostics",
    "CaptumIntegratedGradients",
]
