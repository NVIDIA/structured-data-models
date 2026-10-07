"""Adapter factory for the common tabular and relational benchmark runners."""

from argparse import Namespace
from collections.abc import Sequence

from research.multigpu.placement import install_icl_placement

from sdm.models.base import ICLModel
from sdm.models.ensemble_parallel import EnsembleParallel


def factory(args: Namespace, replicas: Sequence[ICLModel]) -> EnsembleParallel:
    """Install placement on one model and reuse the resident recipe executor.

    Example runner flags: --mode adapter --replicas 1 --gpus 4
    --adapter research.multigpu.icl_placement:factory --placement layers.
    Both baseline and placed runs must use matching member_seed in fit().
    """
    if len(replicas) != 1:
        raise ValueError("Placement requires exactly one model replica")
    devices = [f"cuda:{index}" for index in range(args.gpus)]
    model = replicas[0]
    for inner in model.models.values():
        install_icl_placement(inner, devices, args.placement)
    return EnsembleParallel([model])
