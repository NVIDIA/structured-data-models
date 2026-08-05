"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.attention as torch_attention

from sdm._constants import NaT
from sdm._warnings import warn_once
from sdm.stype import Stype, StypeLike, infer_stypes
from sdm.tensor import (
    VarLenTensor,
    StringTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
)
from sdm.relational import (
    Relationship,
    RelationalData,
    TaskLink,
    RelatedTables,
    TemporalSamplingConfig,
)
from sdm import evaluation, models


def _activate_fa3() -> None:
    activate = getattr(torch_attention, "activate_flash_attention_impl", None)
    current = getattr(torch_attention, "current_flash_attention_impl", None)
    if (
        activate is None
        or current is None
        or current() is not None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability(0)[0] != 9
    ):
        return

    try:
        activate("FA3")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        warn_once(
            "fa3-activation-failed",
            f"FA3 could not be enabled on Hopper ({error}); falling back "
            "to PyTorch's default attention implementation.",
            stacklevel=2,
        )


_activate_fa3()

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "NaT",
    "Stype",
    "StypeLike",
    "infer_stypes",
    "VarLenTensor",
    "StringTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
    "Relationship",
    "RelationalData",
    "TaskLink",
    "RelatedTables",
    "TemporalSamplingConfig",
    "evaluation",
    "models",
    "__version__",
]
