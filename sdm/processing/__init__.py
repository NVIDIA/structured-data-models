"""Fittable pre/postprocessing transforms for structured data."""

from sdm.processing.base import Processor, InvertibleMixin, SharedState
from sdm.processing.sequential import Sequential
from sdm.processing.choice import Choice
from sdm.processing.stype_dispatch import StypeDispatch
from sdm.processing.task_dispatch import TaskDispatch
from sdm.processing.recipe import Recipe
from sdm.processing.numerical import (
    Clip,
    QuantileClip,
    SigmaClip,
    MeanImpute,
    Power,
    Quantile,
    StandardScale,
    ConstantFilter,
)
from sdm.processing.categorical import (
    CategoricalAlign,
    CategoricalImpute,
    CategoryShuffle,
)
from sdm.processing.datetime import EncodeDatetime
from sdm.processing.output import EnsembleReduce, SoftmaxTemperature
from sdm.processing.common import Identity, FeaturePermute, ToNumerical

__all__ = [
    "Processor",
    "InvertibleMixin",
    "SharedState",
    "Sequential",
    "Choice",
    "StypeDispatch",
    "CategoricalAlign",
    "CategoricalImpute",
    "TaskDispatch",
    "Clip",
    "QuantileClip",
    "CategoryShuffle",
    "ConstantFilter",
    "EncodeDatetime",
    "FeaturePermute",
    "Identity",
    "MeanImpute",
    "Power",
    "Quantile",
    "SigmaClip",
    "EnsembleReduce",
    "SoftmaxTemperature",
    "StandardScale",
    "ToNumerical",
    "Recipe",
]
