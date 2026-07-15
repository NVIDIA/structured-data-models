"""Fittable pre/postprocessing transforms for structured data."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.sequential import Sequential
from sdm.processing.choice import Choice
from sdm.processing.stype_dispatch import StypeDispatch
from sdm.processing.categorical_align import CategoricalAlign
from sdm.processing.categorical_impute import CategoricalImpute
from sdm.processing.task_dispatch import TaskDispatch
from sdm.processing.clip import Clip
from sdm.processing.quantile_clip import QuantileClip
from sdm.processing.category_shuffle import CategoryShuffle
from sdm.processing.constant_filter import ConstantFilter
from sdm.processing.feature_permute import FeaturePermute
from sdm.processing.identity import Identity
from sdm.processing.impute import MeanImpute
from sdm.processing.postprocess import SoftmaxTemperature
from sdm.processing.power import Power
from sdm.processing.reduce_estimators import ReduceEstimators
from sdm.processing.quantile import Quantile
from sdm.processing.recipe import Recipe
from sdm.processing.sigma_clip import SigmaClip
from sdm.processing.standard_scale import StandardScale
from sdm.processing.to_numerical import ToNumerical

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
    "FeaturePermute",
    "Identity",
    "MeanImpute",
    "Power",
    "Quantile",
    "SigmaClip",
    "ReduceEstimators",
    "SoftmaxTemperature",
    "StandardScale",
    "ToNumerical",
    "Recipe",
]
