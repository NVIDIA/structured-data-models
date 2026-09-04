"""Cross-stype processors."""

from sdm.processing.common.identity import Identity
from sdm.processing.common.callable import Callable
from sdm.processing.common.sequential import Sequential
from sdm.processing.common.drop import DropStypes
from sdm.processing.common.stype import StypeDispatch
from sdm.processing.common.task import TaskDispatch
from sdm.processing.common.table import TableDispatch
from sdm.processing.common.ensemble import EnsembleProcessorAdapter
from sdm.processing.common.choice import Choice
from sdm.processing.common.to_numerical import ToNumerical
from sdm.processing.common.shuffle import ShuffleColumns
from sdm.processing.common.select import SelectColumns, SelectRows

__all__ = [
    "Identity",
    "Callable",
    "Sequential",
    "DropStypes",
    "StypeDispatch",
    "TaskDispatch",
    "TableDispatch",
    "EnsembleProcessorAdapter",
    "Choice",
    "ToNumerical",
    "ShuffleColumns",
    "SelectColumns",
    "SelectRows",
]
