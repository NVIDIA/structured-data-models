"""Run tabular benchmark jobs with memory reclamation between jobs."""

import gc
from pathlib import Path

import torch
from tabarena.benchmark.experiment import Job
from tabarena.contexts import AbstractArenaContext


def run_jobs(
    context: AbstractArenaContext,
    jobs: list[Job],
    result_dir: Path,
) -> None:
    for job in jobs:
        context.run_jobs(jobs=[job], expname=result_dir, register=False)
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch._C._host_emptyCache()
            torch.cuda.empty_cache()
