# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test TabArena training and evaluation on the bundled Iris dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.model import MODEL_CONFIGS

dataset = load_iris(as_frame=True).frame
train_indices, test_indices = next(
    StratifiedKFold(n_splits=2).split(
        dataset.drop(columns="target"),
        dataset["target"],
    )
)

with TemporaryDirectory() as tmp_dir:
    output_dir = Path(tmp_dir)
    task = UserTask(task_name="iris", task_cache_path=output_dir / "tasks")
    task.save_task(
        task.create_task(
            dataset=dataset,
            target_feature="target",
            problem_type="classification",
            splits={0: {0: (train_indices.tolist(), test_indices.tolist())}},
        )
    )
    context = TabArenaContext(
        methods=[],
        task_metadata=TaskMetadataCollection.from_user_tasks(task),
        backend="native",
        fillna_method=None,
        calibration_method=None,
    )
    generator = ConfigGenerator(
        search_space={},
        model_cls=MODEL_CONFIGS["tabiclv2"].model_cls,
        manual_configs=[{}],
    )
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(generator, 0)],
        outer_experiments=True,
    ).build_experiments()
    results = context.build_and_run_jobs(
        experiments=experiments,
        expname=output_dir / "results",
        user_tasks=[task],
        debug_mode=True,
    )
    assert len(results) == 1
    leaderboard = context.compare(output_dir=output_dir / "evals")
    assert not leaderboard.empty
    print(leaderboard.to_string(index=False))
