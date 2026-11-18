# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activations for neural networks and matrix generation."""

from sdm.prior.activations.base import Activation
from sdm.prior.activations.callable import CallableActivation
from sdm.prior.activations.standardized import (
    StandardizedActivation,
)
from sdm.prior.activations.parametric import (
    ReLUPowActivation,
    SignPowActivation,
    InversePowActivation,
    IntegerPowActivation,
)
from sdm.prior.activations.mixture import MixtureActivation
from sdm.prior.activations.fixed import (
    identity,
    heaviside,
    gaussian,
    unit_indicator,
    softmax,
    one_hot_argmax,
    argsort,
    log_abs,
    rank,
    modulo,
)

__all__ = [
    "Activation",
    "CallableActivation",
    "StandardizedActivation",
    # Parametric:
    "ReLUPowActivation",
    "SignPowActivation",
    "InversePowActivation",
    "IntegerPowActivation",
    # Combos:
    "MixtureActivation",
    # Functions for CallableActivation:
    "identity",
    "heaviside",
    "gaussian",
    "unit_indicator",
    "softmax",
    "one_hot_argmax",
    "argsort",
    "log_abs",
    "rank",
    "modulo",
]
