# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static-shape CUDA graph replay for resident tabular ensemble members."""

from __future__ import annotations

import time
from argparse import Namespace
from collections.abc import Sequence
from functools import partial
from typing import Any

import torch

from sdm import Stype, TableTensor
from sdm.cache import Cache
from sdm.models import EnsembleParallel, ICLModel, KumoTabular
from sdm.processing.execution import MemberQuery


class GraphEnsembleParallel(EnsembleParallel):
    """Capture each fitted member and query shape as a separate CUDA graph.

    Args:
        replicas: Identical evaluation-mode KumoTabular models on CUDA devices.

    Shared recipe processing remains eager. Graphs retain inputs, outputs,
    cache state, and private graph memory pools until clear/close. A new query
    shape triggers a new explicit capture recorded in ``capture_events``.
    """

    def __init__(self, replicas: Sequence[ICLModel]) -> None:
        if any(not isinstance(model, KumoTabular) for model in replicas):
            raise TypeError(
                "CUDA graph adapter currently supports KumoTabular"
            )
        if any(
            next(model.parameters()).device.type != "cuda"
            for model in replicas
        ):
            raise ValueError("CUDA graph replicas require CUDA devices")
        self._graphs: list[dict[Any, Any]] = [{} for _ in replicas]
        self.capture_events: list[dict[str, Any]] = []
        self._originals = [model._forward_batch for model in replicas]
        super().__init__(replicas)

        for worker_id, model in enumerate(replicas):
            model._forward_batch = partial(self._forward_graph, worker_id)

    @property
    def graph_capture_s(self) -> float:
        """Sum capture/warmup durations across workers; these may overlap."""
        return sum(event["capture_s"] for event in self.capture_events)

    @property
    def graph_count(self) -> int:
        """Number of captured member/shape/precision combinations."""
        return sum(len(graphs) for graphs in self._graphs)

    def _forward_graph(
        self, worker_id: int, **kwargs: Any
    ) -> list[TableTensor]:
        queries = kwargs["queries"]
        if len(queries) != 1 or queries[0].related_tables is not None:
            raise ValueError(
                "CUDA graph adapter requires individual tabular members"
            )
        query = queries[0]
        if query.x.active_stypes != {Stype.numerical}:
            raise ValueError(
                "CUDA graph adapter expects numerical preprocessed features"
            )
        device = self.devices[worker_id]
        enabled = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")
        key = (
            id(kwargs["cache"]),
            tuple(query.x.shape),
            query.x.dtype,
            enabled,
            dtype,
        )
        states = self._graphs[worker_id]
        if key not in states:
            start = time.perf_counter()
            static_x = TableTensor(
                columns=query.x.columns, numerical=query.x.numerical.clone()
            )
            graph_kwargs = dict(kwargs)
            graph_kwargs["queries"] = [MemberQuery(static_x, None)]
            cache = kwargs["cache"]
            classes = cache["classes"]
            # Labels only name output columns. Keeping this metadata on CPU
            # avoids the .tolist() device synchronization inside capture.
            graph_kwargs["cache"] = Cache(
                cache, classes=None if classes is None else classes.cpu()
            ).freeze()
            original = self._originals[worker_id]
            graph = torch.cuda.CUDAGraph()
            with torch.autocast(
                "cuda", dtype=dtype, enabled=enabled, cache_enabled=False
            ):
                for _ in range(2):
                    original(**graph_kwargs)
                torch.cuda.current_stream(device).synchronize()
                with torch.cuda.graph(
                    graph,
                    stream=torch.cuda.current_stream(device),
                    capture_error_mode="thread_local",
                ):
                    outputs = original(**graph_kwargs)
            states[key] = (graph, static_x, outputs, graph_kwargs)
            self.capture_events.append(
                {
                    "worker": worker_id,
                    "query_shape": list(query.x.shape),
                    "capture_s": time.perf_counter() - start,
                    "autocast_enabled": enabled,
                    "autocast_dtype": str(dtype),
                }
            )
        graph, static_x, outputs, _ = states[key]
        static_x.numerical.copy_(query.x.numerical)
        graph.replay()
        # Later replays overwrite graph outputs; give the caller owned tensors.
        return [output.clone() for output in outputs]

    def clear(self) -> None:
        """Drop all graphs before releasing the corresponding member caches."""
        for states in self._graphs:
            states.clear()
        self.capture_events = []
        super().clear()

    def close(self) -> None:
        """Join workers, release captures, and restore replica methods."""
        super().close()
        for model, original in zip(
            self.replicas, self._originals, strict=True
        ):
            model._forward_batch = original


def factory(
    args: Namespace, replicas: Sequence[ICLModel]
) -> GraphEnsembleParallel:
    """Build graph replay workers for the shared tabular benchmark."""
    model = GraphEnsembleParallel(replicas)
    model.fit = partial(model.fit, member_seed=args.seed)
    return model
