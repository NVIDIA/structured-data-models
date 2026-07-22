"""Fittable pre/postprocessing transforms for structured data."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.sequential import Sequential
from sdm.processing.choice import Choice
from sdm.processing.stype_dispatch import StypeDispatch
from sdm.processing.task_dispatch import TaskDispatch
from sdm.processing.recipe import Recipe
from sdm.processing.numerical.clip import Clip
from sdm.processing.numerical.quantile_clip import QuantileClip
from sdm.processing.numerical.sigma_clip import SigmaClip
from sdm.processing.numerical.impute import MeanImpute
from sdm.processing.numerical.power import Power
from sdm.processing.numerical.quantile import Quantile
from sdm.processing.numerical.standard_scale import StandardScale
from sdm.processing.numerical.constant_filter import ConstantFilter
from sdm.processing.categorical.categorical_align import CategoricalAlign
from sdm.processing.categorical.categorical_impute import CategoricalImpute
from sdm.processing.categorical.category_shuffle import CategoryShuffle
from sdm.processing.datetime.datetime import EncodeDatetime
from sdm.processing.output.ensemble_reduce import EnsembleReduce
from sdm.processing.output.postprocess import SoftmaxTemperature
from sdm.processing.cross_stype.identity import Identity
from sdm.processing.cross_stype.feature_permute import FeaturePermute
from sdm.processing.cross_stype.to_numerical import ToNumerical

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
    "ToNumerical",
    "Recipe",
]
