# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, TableTensor, Task
from sdm.prior import HP, Correlated, CorrelatedContext, DAGSampler, LogUniform
from sdm.prior.converters import (
    CategoricalConverter,
    MixtureConverter,
    NumericalConverter,
)
from sdm.prior.functions import ParentMerger, RandomFunction
from sdm.prior.root_samplers import RootSampler
from sdm.prior.tabular.schema import Schema, SchemaSampler
from sdm.prior.tabular.transform.base import PriorTransform
from sdm.prior.util import random_weight, standardize


class SCMPrior:
    """Sample datasets from random structural causal models (SCMs).

    Sample a directed acyclic graph and evaluate random functions from
    root to leaf. Nodes carry latent channels; converters map selected
    channels to feature and target columns and may update the values
    passed to child nodes. Apply dataset transforms to the resulting tables.

    Args:
        schema_sampler: Dataset schema sampler.
        dag_sampler: DAG sampler.
        latent_channels: Number of unobserved channels per node, in addition
            to channels assigned to feature and target columns.
        root_sampler: Sampler for initial values at nodes without parents.
        random_fn: Random function for computing node values.
        parent_merger: Rule for computing node values from multiple parents
            using ``random_fn``.
        numerical_converter: Converter to numerical column values, including
            regression targets. A mixture selects a converter per column.
        categorical_converter: Converter to category indices, including
            classification targets. A mixture selects a converter per column.
        transforms: Ordered dataset transforms after column conversion.
    """

    def __init__(
        self,
        schema_sampler: SchemaSampler,
        dag_sampler: DAGSampler,
        latent_channels: HP[int] | int,
        root_sampler: RootSampler,
        random_fn: RandomFunction,
        parent_merger: ParentMerger,
        numerical_converter: NumericalConverter
        | MixtureConverter[NumericalConverter],
        categorical_converter: CategoricalConverter
        | MixtureConverter[CategoricalConverter],
        transforms: Sequence[PriorTransform] = (),
    ) -> None:
        self.schema_sampler = schema_sampler
        self.dag_sampler = dag_sampler
        self.latent_channels = HP.from_value(latent_channels)
        self.root_sampler = root_sampler
        self.random_fn = random_fn
        self.parent_merger = parent_merger
        self.numerical_converter = numerical_converter
        self.categorical_converter = categorical_converter
        self.transforms = tuple(transforms)

    @torch.no_grad()
    def sample(
        self,
        num_rows: int,
        *,
        schema: Schema | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor] | None:
        """Sample a synthetic dataset.

        Args:
            num_rows: Number of rows in the dataset.
            schema: Schema to use instead of drawing from the schema sampler.
            generator: Pseudorandom generator.

        Returns:
            Features of shape ``[R, C]`` and targets of shape ``[R, 1]``,
            where ``R`` is the row count and ``C`` is the feature count.
            Classification targets are categorical; regression targets are
            numerical. Return ``None`` if generation or a transform rejects
            the dataset.
        """
        if schema is None:
            schema = self.schema_sampler.sample(generator=generator)
        dag = self.dag_sampler.sample(generator=generator)
        dag = dag.with_columns(schema.num_columns, generator=generator)

        # Propagate node values through the DAG.
        with CorrelatedContext(generator=generator):
            node_channels = self.latent_channels.sample_batch(
                size=dag.num_nodes,
                generator=generator,
            ).tolist()

            column_nodes = [
                cast(int, dag.target_node),
                *cast(Tensor, dag.input_nodes).tolist(),
            ]
            column_dims = [schema.target_dim, *schema.column_dims]

            numerical_columns = [
                i for i, dim in enumerate(column_dims) if dim == 1
            ]
            categorical_columns = [
                i for i, dim in enumerate(column_dims) if dim > 1
            ]
            feature_dims = column_dims.copy()
            columns_per_node: list[
                dict[NumericalConverter | CategoricalConverter, list[int]]
            ] = [{} for _ in range(dag.num_nodes)]
            # Resolve mixtures once, before allocating each column's channels.
            for configured, columns in (
                (self.numerical_converter, numerical_columns),
                (self.categorical_converter, categorical_columns),
            ):
                if isinstance(configured, MixtureConverter):
                    converters = configured.sample(
                        num_columns=len(columns), generator=generator
                    )
                else:
                    converters = [configured] * len(columns)
                groups: dict[
                    NumericalConverter | CategoricalConverter, list[int]
                ] = {}
                for column, converter in zip(columns, converters):
                    groups.setdefault(converter, []).append(column)
                for converter, columns in groups.items():
                    if isinstance(converter, NumericalConverter):
                        sampled_dims = converter.sample_num_channels(
                            num_columns=len(columns), generator=generator
                        ).tolist()
                        for column, dim in zip(columns, sampled_dims):
                            feature_dims[column] = dim
                    for column in columns:
                        columns_per_node[column_nodes[column]].setdefault(
                            converter, []
                        ).append(column)

            column_offset: list[int] = []
            for i, dim in zip(column_nodes, feature_dims):
                column_offset.append(node_channels[i])
                node_channels[i] += dim

            node_values: list[Tensor | None] = [None] * dag.num_nodes
            column_values: list[Tensor | None] = [None] * len(column_nodes)
            node_scale = Correlated(LogUniform(0.1, 10.0))

            for i in dag.pruned_topological_order.tolist():
                parents = dag.parents(i).tolist()
                if len(parents) == 0:
                    x = self.root_sampler.sample(
                        num_rows=num_rows,
                        channels=node_channels[i],
                        generator=generator,
                    )
                    x = self.random_fn.forward(
                        x=x,
                        out_channels=node_channels[i],
                        generator=generator,
                    )
                elif len(parents) == 1:
                    x = self.random_fn.forward(
                        x=node_values[parents[0]],
                        out_channels=node_channels[i],
                        generator=generator,
                    )
                else:
                    x = self.parent_merger.forward(
                        parent_values=[node_values[j] for j in parents],
                        random_fn=self.random_fn,
                        out_channels=node_channels[i],
                        generator=generator,
                    )

                # Standardize channels and scale each by a random weight.
                x, _ = standardize(x, dim=-2)
                weights = random_weight(
                    size=(x.size(-1),),
                    dtype=x.dtype,
                    generator=generator,
                )
                x = x * weights

                # Normalize by the RMS row norm.
                rms_norm = (
                    x.square()
                    .sum(dim=-1, keepdim=True)
                    .mean(dim=-2, keepdim=True)
                )
                x = x / (rms_norm.sqrt() + 1e-8)

                for converter, columns in columns_per_node[i].items():
                    slices = [
                        x[
                            ...,
                            column_offset[c] : column_offset[c]
                            + feature_dims[c],
                        ].T
                        for c in columns
                    ]  # each: [channels, num_rows]
                    packed = torch.nested.as_nested_tensor(
                        slices, layout=torch.jagged
                    )
                    values, child_slices = converter.forward(
                        slices=packed, generator=generator
                    )

                    # Store columns and update slices passed to child nodes.
                    for c, value, column in zip(
                        columns, values, child_slices.unbind()
                    ):
                        column_values[c] = value
                        start = column_offset[c]
                        x[..., start : start + feature_dims[c]] = column.T

                scale = node_scale.sample(generator=generator)
                node_values[i] = x * scale

            x = torch.stack(
                tensors=[
                    c.to(torch.get_default_dtype())
                    for c in column_values[1:]
                    if c is not None
                ],
                dim=-1,
            )  # [num_rows, num_columns]

            y = column_values[0]
            assert y is not None

        y = y.unsqueeze(-1)
        numerical = [i for i, dim in enumerate(schema.column_dims) if dim == 1]
        categorical = [
            i for i, dim in enumerate(schema.column_dims) if dim > 1
        ]
        if not self.transforms:
            if not (x.isfinite().all() & y.isfinite().all()):
                return None
        elif not x[:, categorical].isfinite().all() or (
            schema.task_type == Task.classification and not y.isfinite().all()
        ):
            return None

        features = TableTensor(
            columns={
                "numerical": tuple(f"x_{i}" for i in numerical),
                "categorical": tuple(f"x_{i}" for i in categorical),
            },
            numerical=x[:, numerical],
            categorical=CategoricalTensor(
                code=x[:, categorical].long(),
                categories=tuple(
                    torch.arange(schema.column_dims[i], device="cpu")
                    for i in categorical
                ),
            ),
        )
        if schema.task_type == Task.regression:
            target = TableTensor(
                columns={"numerical": ("target",)}, numerical=y
            )
        else:
            target = TableTensor(
                columns={"categorical": ("target",)},
                categorical=CategoricalTensor(
                    code=y,
                    categories=(
                        torch.arange(schema.target_dim, device="cpu"),
                    ),
                ),
            )
        for transform in self.transforms:
            result = transform(x=features, y=target, generator=generator)
            if result is None:
                return None
            features, target = result
        if not target.numerical.isfinite().all():
            return None
        return features, target
