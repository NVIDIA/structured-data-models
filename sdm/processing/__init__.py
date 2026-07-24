"""Fittable pre/postprocessing transforms for structured data."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.common import (
    Sequential,
    Choice,
    StypeDispatch,
    TaskDispatch,
    Identity,
    FeaturePermute,
    ToNumerical,
)
from sdm.processing.text.tfidf_transformer import TfidfTransformer
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
from sdm.processing.recipe import Recipe

__all__ = [
    "Processor",
    "InvertibleMixin",
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
    "TfidfTransformer",
    "ToNumerical",
    "Recipe",
]
