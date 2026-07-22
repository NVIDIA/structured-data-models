"""Processors that are stype-agnostic or bridge multiple stypes."""

from sdm.processing.cross_stype.constant_filter import ConstantFilter
from sdm.processing.cross_stype.feature_permute import FeaturePermute
from sdm.processing.cross_stype.identity import Identity
from sdm.processing.cross_stype.tfidf_encoder import TfidfEncoder
from sdm.processing.cross_stype.to_numerical import ToNumerical

__all__ = [
    "ConstantFilter",
    "FeaturePermute",
    "Identity",
    "TfidfEncoder",
    "ToNumerical",
]
