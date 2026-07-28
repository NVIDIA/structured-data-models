r"""Run TabICLv2 on TabArena.

$ uv run --group example-tabarena python examples/tabiclv2/tabarena/main.py

The output directory must be empty.
"""

from __future__ import annotations

from pathlib import Path

from model import SDMTabICLv2System
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import SystemConfigGenerator

result_dir = Path(__file__).parent.parent / "tabarena_out" / "TabICLv2"
if result_dir.exists() and any(result_dir.iterdir()):
    raise FileExistsError(
        f"Result directory {result_dir} is non-empty. Choose a fresh path."
    )
result_dir.mkdir(parents=True, exist_ok=True)

generator = SystemConfigGenerator(
    model_cls=SDMTabICLv2System,
    name="SDMTabICLv2System",
    manual_configs=[{}],
)
experiments = TabArenaV0pt1ExperimentBundle(
    models=[(generator, 0)],
    system_experiments=True,
).build_experiments()

context = TabArenaContext()
context.build_and_run_jobs(
    experiments,
    expname=result_dir,
    register=False,
    debug_mode=True,
)
