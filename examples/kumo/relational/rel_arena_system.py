r"""Kumo system submission with validation-owned recipe selection.

Unlike the MODEL example, this system owns tuning and can score candidates on
a fixed validation subset. It refits its winner on full outer TRAIN+VAL and
returns predictions for the entire masked TEST table. It compares text OFF
against context-fitted PCA32 with all other parameters fixed, using the same
rel_arena_search_space.py policy as the model.

Run one task from the repository root, after the setup in README.relarena.md::

    python -m examples.kumo.relational.rel_arena_system \
        --datasets rel-f1 --tasks driver-position \
        --search-space examples/kumo/relational/rel_arena_search_space.py \
        --output system-results.csv

--search-space is optional; omit it to use the bundled policy. Custom Python
files must export SEARCH_SPACE with a fixed_grid; this system evaluates every
entry. Loading executes the file, so use only trusted files.

The CLI uses full validation. To choose a validation subset, call Python::

    from pathlib import Path

    from examples.kumo.relational.rel_arena_system import KumoSystem
    from relarena.cache import CacheConfig
    from relarena.dataset import RelBenchDatasetTask

    source = RelBenchDatasetTask("rel-f1", "driver-position")
    system = KumoSystem(
        cache=CacheConfig(
            directory=Path(".cache/relarena/rel-f1-driver-position"),
            on_miss="fill",
        ),
    )
    predictions = system.run(
        source.task,
        inner_split=source.inner_split(),
        outer_split=source.outer_split(),
        seed=0,
        validation_rows=10_000,
    )
    print(source.task.evaluate(predictions))

Set cache_dir to an empty directory to cache newly computed Qwen embeddings
across tuning, refitting, and test prediction. No precomputation is required.
Omit it to disable persistent caching. PCA remains fitted separately on each
training context. Alternatively, point cache_dir at precompute_text.py's output
directory to reuse existing embeddings; missing documents are encoded and
cached.
run_system_experiment accepts cache_dir; when constructing KumoSystem directly,
set CacheConfig.directory as shown above.

Omit validation_rows (or pass None) for full validation. This keyword belongs
to KumoSystem.run; RelArena's run_system_experiment does not forward it.
Every candidate uses the same seed-selected rows, sampled without labels.
Tiny binary samples may lack a class and make validation scoring fail.
An optional time_limit covers this run; account for earlier preprocessing
separately when checking the complete per-task runtime allowance.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace

import numpy as np
from examples.kumo.relational.rel_arena_model import KumoPredictor
from examples.kumo.relational.rel_arena_search_space import (
    SEARCH_SPACE,
    parse_search_space,
)
from relarena.dataset import InnerSplit, OuterSplit, concat_tables
from relarena.metrics import primary_metric
from relarena.registry import register_system
from relarena.runner import select_best
from relarena.system import RelArenaSystem
from relarena.tuner import plan_configs, run_trial
from relbench.base import EntityTask, Table


@register_system
class KumoSystem(RelArenaSystem):
    """Select on validation, then refit only the winning recipe."""

    name = "sdm-kumo-system"

    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
        validation_rows: int | None = None,
    ) -> np.ndarray:
        """Return aligned outer predictions within one soft procedure budget.

        The caller is responsible for including data preparation before this
        method receives the splits in the complete method budget.

        Args:
            task: RelBench task defining the prediction target and metric.
            inner_split: Training data and validation-censored database.
            outer_split: Final-fit data and masked TEST queries.
            seed: Seed for validation sampling and every candidate fit.
            time_limit: Soft budget in seconds for selection, refit and
                prediction.
            validation_rows: Maximum validation rows per candidate, or None for
                all rows. One uniform sample without replacement is reused for
                every candidate; full TRAIN+VAL and TEST remain unchanged.

        The actual validation count is available as ``self.validation_rows``.
        """
        deadline = (
            None if time_limit is None else time.monotonic() + time_limit
        )
        if validation_rows is not None and validation_rows < len(
            inner_split.eval_table
        ):
            rows = np.sort(
                np.random.default_rng(seed).choice(
                    len(inner_split.eval_table),
                    size=validation_rows,
                    replace=False,
                )
            )
            eval_table, eval_target = [
                Table(
                    df=table.df.iloc[rows].reset_index(drop=True),
                    fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                    pkey_col=table.pkey_col,
                    time_col=table.time_col,
                )
                for table in (inner_split.eval_table, inner_split.eval_target)
            ]
            inner_split = replace(
                inner_split, eval_table=eval_table, eval_target=eval_target
            )
        self.validation_rows = len(inner_split.eval_table)

        def remaining() -> float | None:
            if deadline is None:
                return None
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("System selection/refit budget exhausted")
            return seconds

        trials = []
        assert SEARCH_SPACE.fixed_grid is not None
        for tag, config in plan_configs(
            SEARCH_SPACE, len(SEARCH_SPACE.fixed_grid), seed
        ):
            trial = run_trial(
                KumoPredictor,
                config,
                tag,
                task,
                inner_split,
                seed=seed,
                time_limit=remaining(),
                cache_predictions=False,
                cache=self.cache,
                run_identity=self.run_identity,
            )
            remaining()
            if (
                not trial.ok
                or trial.val_score is None
                or not math.isfinite(trial.val_score)
            ):
                raise RuntimeError(
                    "System candidate did not complete validation"
                )
            trials.append(trial)

        selected = select_best(trials, primary_metric(task))
        self.selected_config = dict(selected.config)
        self.validation_scores = [trial.val_score for trial in trials]
        predictor = KumoPredictor(
            selected.config, cache=self.cache, run_identity=self.run_identity
        )
        predictor.fit(
            task,
            outer_split.db_state,
            concat_tables(outer_split.train_table, outer_split.val_table),
            None,
            seed=seed,
            time_limit=remaining(),
        )
        remaining()
        predictions = predictor.predict(
            task, outer_split.db_state, outer_split.eval_table
        )
        remaining()
        return predictions


if __name__ == "__main__":
    from relarena.cli import main

    SEARCH_SPACE, arguments = parse_search_space(sys.argv[1:])
    raise SystemExit(main(["--model", KumoSystem.name, *arguments]))
