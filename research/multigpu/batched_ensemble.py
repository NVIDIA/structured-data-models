# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Research adapter combining device distribution and local member batching."""

from __future__ import annotations

import copy
from argparse import Namespace
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import torch
from torch import Tensor

from sdm import EnsembleTable, Recipe, RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models import EnsembleParallel, ICLModel, KumoTabular
from sdm.models.base import (
    _batch_slices,
    _categorical_mask,
    _class_values,
    _stack_context,
)
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)


def factory(
    args: Namespace, replicas: Sequence[ICLModel]
) -> BatchedEnsembleParallel:
    """Construct the adapter for ``tabular_bench --mode adapter``."""
    model = BatchedEnsembleParallel(replicas)
    model.fit = partial(
        model.fit,
        member_seed=args.seed,
        estimator_batch_size=args.estimator_batch_size,
    )
    return model


class BatchedEnsembleParallel(EnsembleParallel):
    """Batch deterministic KumoTabular members before distributing batches.

    Pass ``estimator_batch_size`` to fit. Members with more than ten classes,
    or models other than KumoTabular, fall back to individual execution so
    their per-member model RNG sequences remain placement independent.
    This research adapter uses existing private batching helpers deliberately;
    it is not yet a second supported public executor implementation.
    """

    def fit(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        y: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int | None = None,
        estimator_batch_size: int | None = 2,
        generator: torch.Generator | None = None,
        member_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Fit compatible member batches and retain caches on assigned GPUs."""
        self.clear()
        execution = RecipeExecution(
            self.replicas[0].default_recipe()
            if recipe is None
            else copy.deepcopy(recipe)
        )
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            contexts = execution.fit_transform(
                x=x,
                y=y,
                related_tables=related_tables,
                num_members=num_estimators,
                generator=generator,
            )
            per_replica = (len(contexts) + len(self.replicas) - 1) // len(
                self.replicas
            )
            estimator_batch_size = min(
                per_replica,
                per_replica
                if estimator_batch_size is None
                else estimator_batch_size,
            )
            # ECOC samples a separate codebook per member. A batched generator
            # cannot reproduce independently seeded member generators.
            if not isinstance(self.replicas[0], KumoTabular) or any(
                context.y.categorical.size(-1) > 0
                and context.y.categorical.categories[0].numel() > 10
                for context in contexts
            ):
                estimator_batch_size = 1
            class_values = _class_values(contexts, estimator_batch_size)
            batches = _batch_slices(
                contexts=contexts,
                queries=None,
                class_values=class_values,
                estimator_batch_size=estimator_batch_size,
            )
        seeds = tuple(member_seed + i for i in range(len(contexts)))

        def fit_batch(i: int, model: ICLModel, device: torch.device) -> Cache:
            batch = batches[i]
            members = []
            for context in contexts[batch]:
                context = MemberContext(
                    x=context.x.to(device, non_blocking=True),
                    y=context.y.to(device, non_blocking=True),
                    related_tables=(
                        None
                        if context.related_tables is None
                        else context.related_tables.to(
                            device, non_blocking=True
                        )
                    ),
                    input_stypes=context.input_stypes,
                )
                members.append(model._prepare_context(context, ()))
            context = _stack_context(members)
            mask = _categorical_mask(members)
            cache = Cache(
                member_ids=tuple(range(len(contexts)))[batch],
                x_schemas=tuple(member.x.schema for member in members),
                related_tables_schema=(
                    None
                    if context.related_tables is None
                    else context.related_tables.schema
                ),
                classes=(
                    context.y.categorical.categories[0]
                    if context.y.categorical.size(-1)
                    else None
                ),
                class_values=class_values[batch],
                categorical_mask=mask,
            )
            model._forward(
                x_context=context.x,
                y_context=context.y,
                x_query=None,
                related_context_tables=context.related_tables,
                related_query_tables=None,
                cache=cache,
                generator=torch.Generator(device=device).manual_seed(
                    seeds[batch.start]
                ),
                categorical_mask=mask,
                **kwargs,
            )
            return cache.freeze()

        self._caches = self._dispatch(fit_batch, len(batches), x.device)
        self._execution = execution
        self._kwargs = kwargs
        self.member_seeds = seeds

    def predict(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:
        """Replay fitted batches and restore original member output order."""
        if self._execution is None:
            raise RuntimeError("Call fit before predict")
        schema = self._caches[0]["related_tables_schema"]
        if related_tables is not None and schema is not None:
            related_tables = related_tables.select_tables(tables=schema.tables)
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            queries = self._execution.transform(x, related_tables)

        def predict_batch(
            i: int, model: ICLModel, device: torch.device
        ) -> list[tuple[int, TableTensor]]:
            cache = self._caches[i]
            member_ids = cast(tuple[int, ...], cache["member_ids"])
            members = []
            for member_id, x_schema in zip(
                member_ids, cache["x_schemas"], strict=True
            ):
                query = queries[member_id]
                query = MemberQuery(
                    x=query.x.to(device, non_blocking=True),
                    related_tables=(
                        None
                        if query.related_tables is None
                        else query.related_tables.to(device, non_blocking=True)
                    ),
                )
                members.append(
                    model._prepare_query(
                        query=query,
                        x_schema=x_schema,
                        related_tables_schema=cache["related_tables_schema"],
                        callbacks=(),
                    )
                )
            outputs = model._forward_batch(
                contexts=None,
                queries=members,
                cache=cache,
                categorical_mask=cache["categorical_mask"],
                class_values=cache["class_values"],
                callbacks=(),
                requires_grad=False,
                generator=None,
                **self._kwargs,
            )
            return [
                (member_id, output.to(x.device, non_blocking=True))
                for member_id, output in zip(member_ids, outputs, strict=True)
            ]

        groups = self._dispatch(predict_batch, len(self._caches), x.device)
        outputs = [
            output
            for _, output in sorted(
                (member for group in groups for member in group),
                key=lambda pair: pair[0],
            )
        ]
        if x.device.type == "cuda":
            for output in outputs:
                output.record_stream(torch.cuda.current_stream(x.device))
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            if self._caches[0]["classes"] is None:
                outputs = list(
                    self._execution.inverse_transform_target(outputs)
                )
            return self._execution.transform_output(outputs)
