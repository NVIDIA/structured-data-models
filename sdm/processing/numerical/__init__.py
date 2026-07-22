"""Processors for numerical (continuous) features."""

from sdm.processing.numerical.clip import Clip
from sdm.processing.numerical.impute import MeanImpute
from sdm.processing.numerical.power import Power
from sdm.processing.numerical.quantile import Quantile
from sdm.processing.numerical.quantile_clip import QuantileClip
from sdm.processing.numerical.sigma_clip import SigmaClip
from sdm.processing.numerical.standard_scale import StandardScale

__all__ = [
    "Clip",
    "MeanImpute",
    "Power",
    "Quantile",
    "QuantileClip",
    "SigmaClip",
    "StandardScale",
]
