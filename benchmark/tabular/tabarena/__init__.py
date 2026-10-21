# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TabArena benchmarks on the official suite or a cached local task."""

from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts import TabArenaContext


def load_context(local_task: str | None = None) -> TabArenaContext:
    """Load the official suite or a cached local ``UserTask``."""
    if local_task is None:
        return TabArenaContext()
    return TabArenaContext(
        methods=[],
        task_metadata=TaskMetadataCollection.from_user_tasks(
            UserTask(task_name=local_task),
        ),
        backend="native",
        fillna_method=None,
        calibration_method=None,
    )
