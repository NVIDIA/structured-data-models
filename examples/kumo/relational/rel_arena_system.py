"""Kumo system submission with validation-owned recipe selection.

Unlike the MODEL example, whose selection is driven by the harness, this
system evaluates the shared candidate policy on the complete inner validation
split, then refits its winner on outer TRAIN+VAL. It returns masked outer
predictions; only RelArena evaluates TEST labels. No task-specific test winners
are used. The policy is provisional, not an optimized submission claim.
"""

from __future__ import annotations

import math
import time

import numpy as np
from examples.kumo.relational._relarena.adapter import KumoPredictor
from examples.kumo.relational._relarena.search_space import SEARCH_SPACE
from relarena.dataset import InnerSplit, OuterSplit, concat_tables
from relarena.metrics import primary_metric
from relarena.registry import register_system
from relarena.runner import select_best
from relarena.system import RelArenaSystem
from relarena.tuner import plan_configs, run_trial
from relbench.base import EntityTask


@register_system
class KumoSystem(RelArenaSystem):
    """Select by full validation, then refit only the winning recipe."""

    name = "sdm-kumo-system"

    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
    ) -> np.ndarray:
        """Return aligned outer predictions within one soft procedure budget.

        The caller is responsible for including data preparation before this
        method receives the splits in the complete method budget.
        """
        deadline = (
            None if time_limit is None else time.monotonic() + time_limit
        )

        def remaining() -> float | None:
            if deadline is None:
                return None
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("System selection/refit budget exhausted")
            return seconds

        trials = []
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
