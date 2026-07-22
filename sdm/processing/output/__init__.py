"""Processors for model output postprocessing."""

from sdm.processing.output.ensemble_reduce import EnsembleReduce
from sdm.processing.output.postprocess import SoftmaxTemperature

__all__ = [
    "EnsembleReduce",
    "SoftmaxTemperature",
]
