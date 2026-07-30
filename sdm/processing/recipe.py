import copy
from collections.abc import Iterable, Sequence
from typing import cast

import torch
from typing_extensions import Self

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.common.sequential import Sequential
from sdm.processing.common.task import TaskDispatch
from sdm.processing.ensemble import (
    EnsembleFitContext,
    EnsemblePlanner,
    EnsembleProcessor,
    EnsembleRelatedTables,
    EnsembleTable,
    VariableSchemaBatchMixin,
    _stack_physical,
    as_ensemble_processor,
)
from sdm.processing.output.reduce import ReduceEstimators
from sdm.processing.output.target import TargetDecode
from sdm.relational import RelatedTables
from sdm.stype import Stype
from sdm.tensor import TableTensor


class _TaskResolver(Processor, InvertibleMixin):
    """Resolve linked output dispatchers while fitting a recipe target.

    The wrapped target processor is a registered child module. Output
    dispatchers stay in a plain tuple so they remain registered only under
    ``Recipe.output``.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        processor: Processor,
        task_dispatchers: tuple[TaskDispatch, ...],
    ) -> None:
        super().__init__()
        self.processor = processor
        self._task_dispatchers = task_dispatchers

    def fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        self.fit_transform(table, generator=generator)
        return self

    def fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self._check_supported_stypes(table)
        self._fitted = False
        for task_dispatcher in self._task_dispatchers:
            task_dispatcher._reset()

        succeeded = False
        try:
            target = self.processor.fit_transform(
                table,
                generator=generator,
            )
            for task_dispatcher in self._task_dispatchers:
                task_dispatcher._resolve(target)
            self._fitted = True
            succeeded = True
            return target
        finally:
            if not succeeded:
                for task_dispatcher in self._task_dispatchers:
                    task_dispatcher._reset()

    def _transform(self, table: TableTensor) -> TableTensor:
        return self.processor.transform(table)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        fn = getattr(self.processor, "inverse_transform", None)
        if not callable(fn):
            raise AttributeError(
                f"{self.processor.__class__.__name__!r} object has no "
                "attribute 'inverse_transform'"
            )
        return fn(table)

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)


class Recipe(torch.nn.Module):
    """Processing contract around an external model boundary.

    A recipe bundles three processing pipelines, one per role the data plays
    relative to the model:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels transformed forward before the model. Regression
      predictions are inverted through this pipeline; classification outputs
      are reconstructed from the fitted target categories instead.
    - ``output``: transforms member outputs after they have been mapped to a
      common class or target space and stacked as ``[E, ..., R, O]``. An
      explicit dimension-changing step such as
      :class:`~sdm.processing.ReduceEstimators` removes ``E``; without one,
      the output remains stacked. Steps before the reducer must support
      stacked outputs, while steps after it receive already-reduced outputs.

    Each configured pipeline remains a normal Processor and can be fitted and
    called independently. Ensemble :meth:`fit_transform` instead fits private
    execution trees; reuse those through :meth:`transform` and
    :meth:`transform_output`. Recipes do not infer each step's non-finite input
    contract; order steps so values are imputed before processors that do not
    explicitly document non-finite support. When ``output`` contains
    :class:`~sdm.processing.TaskDispatch`, fitting ``target`` also selects its
    task-specific output route.

    Copy a task-aware recipe as a whole so its target remains connected to the
    output dispatchers.

    Args:
        features: Steps applied to model inputs before the model.
        target: Steps applied to labels. Invertible numerical target steps map
            regression output back to the original space.
        output: Steps applied to stacked member outputs after member-local
            mappings. Estimator reduction, when desired, is an explicit step
            in this pipeline.
        ensemble_planner: Optional coordinator for member decisions spanning
            more than one Processor path.
    """

    features: Processor
    target: Processor
    output: Processor

    def __init__(
        self,
        features: Processor | Iterable[Processor] | None = None,
        target: Processor | Iterable[Processor] | None = None,
        output: Processor | Iterable[Processor] | None = None,
        *,
        ensemble_planner: EnsemblePlanner | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = Sequential()
        elif not isinstance(features, Processor):
            features = Sequential(*features)

        if target is None:
            target = Sequential()
        elif not isinstance(target, Processor):
            target = Sequential(*target)

        if output is None:
            output = Sequential()
        elif not isinstance(output, Processor):
            output = Sequential(*output)

        # TODO: Support TaskDispatch in features after defining task-aware
        # feature fit ordering.
        for role, processor in (
            ("features", features),
            ("target", target),
        ):
            if any(
                isinstance(module, TaskDispatch)
                for module in processor.modules()
            ):
                raise ValueError(
                    f"'TaskDispatch' is only supported in 'Recipe.output' "
                    f"(found in {role!r})."
                )

        # Common output steps can remain adjacent; nesting would require
        # defining whether dispatchers in inactive branches are resolved.
        task_dispatch_entries = tuple(
            (path, module)
            for path, module in output.named_modules(remove_duplicate=False)
            if isinstance(module, TaskDispatch)
        )
        if isinstance(output, TaskDispatch):
            direct_paths = {""}
        elif isinstance(output, Sequential):
            direct_paths = {
                str(index)
                for index, step in enumerate(output)
                if isinstance(step, TaskDispatch)
            }
        else:
            direct_paths = set()

        nested_paths = tuple(
            path
            for path, _ in task_dispatch_entries
            if path not in direct_paths
        )
        if len(nested_paths) > 0:
            locations = ", ".join(repr(path) for path in nested_paths)
            raise ValueError(
                "'TaskDispatch' must be a direct step in 'Recipe.output'; "
                f"nested task dispatch was found at {locations}."
            )

        task_dispatchers = tuple(
            module
            for path, module in task_dispatch_entries
            if path in direct_paths
        )

        if len(task_dispatchers) > 0:
            target = _TaskResolver(
                processor=target,
                task_dispatchers=task_dispatchers,
            )

        self.features = features
        self.target = target
        self.output = output
        self._ensemble_features: EnsembleProcessor | None = None
        self.ensemble_planner = ensemble_planner
        self._ensemble_related = torch.nn.ModuleList()
        self._related_table_names: tuple[str, ...] = ()
        self._ensemble_output: Processor | None = None
        self._num_members = 0

    @staticmethod
    def _steps(processor: Processor) -> tuple[Processor, ...]:
        if isinstance(processor, Sequential):
            return tuple(processor)
        return (processor,)

    def _validate_ensemble_composition(self) -> None:
        for role, processor in (
            ("features", self.features),
            ("target", self.target),
        ):
            if any(
                isinstance(module, ReduceEstimators)
                for module in processor.modules()
            ):
                raise ValueError(
                    "'ReduceEstimators' is only supported in Recipe.output "
                    f"(found in {role!r})."
                )

        steps = self._steps(self.output)

        direct_reducers = tuple(
            index
            for index, step in enumerate(steps)
            if isinstance(step, ReduceEstimators)
        )
        all_reducers = tuple(
            module
            for module in self.output.modules()
            if isinstance(module, ReduceEstimators)
        )
        if len(all_reducers) > len(direct_reducers):
            raise ValueError(
                "'ReduceEstimators' must be a direct Recipe.output step."
            )
        if len(direct_reducers) > 1:
            raise ValueError(
                "Recipe.output supports at most one 'ReduceEstimators'."
            )

        direct_decoders = tuple(
            index
            for index, step in enumerate(steps)
            if isinstance(step, TargetDecode)
        )
        all_decoders = tuple(
            module
            for module in self.output.modules()
            if isinstance(module, TargetDecode)
        )
        if len(all_decoders) > len(direct_decoders):
            raise ValueError(
                "'TargetDecode' must be a direct Recipe.output step."
            )
        if len(direct_decoders) > 1:
            raise ValueError(
                "Recipe.output supports at most one 'TargetDecode'."
            )
        fitted_output_node = next(
            (
                (path, module)
                for path, module in self.output.named_modules(
                    remove_duplicate=False
                )
                if isinstance(module, Processor)
                and module.requires_fit
                and not isinstance(module, Sequential)
            ),
            None,
        )
        if fitted_output_node is not None:
            path, module = fitted_output_node
            location = path or "<root>"
            raise ValueError(
                "Recipe.output supports only stateless processors; "
                f"{module.__class__.__name__!r} at {location!r} requires "
                "fit."
            )

        if len(direct_reducers) == 0:
            return

        reducer = direct_reducers[0]
        if any(
            isinstance(step, TargetDecode) for step in steps[reducer + 1 :]
        ):
            raise ValueError(
                "'TargetDecode' must run before 'ReduceEstimators'."
            )
        if len(direct_decoders) == 0:
            raise ValueError(
                "'TargetDecode' is required before 'ReduceEstimators'."
            )

    @staticmethod
    def _target_processor(processor: Processor) -> Processor:
        if isinstance(processor, _TaskResolver):
            return processor.processor
        return processor

    @staticmethod
    def _task_from_target(table: TableTensor) -> str:
        if table.size(-1) != 1:
            raise ValueError(
                "Expected the transformed target to contain exactly one "
                f"column (got {table.size(-1)} columns)."
            )
        if table.numerical.size(-1) == 1:
            return "regression"
        if table.categorical.size(-1) == 1:
            return "classification"
        raise ValueError(
            "Expected the transformed target to be numerical or categorical."
        )

    @staticmethod
    def _validate_variable_schema_input(
        *,
        role: str,
        table: TableTensor,
        processor: Processor,
    ) -> None:
        if table.dim() == 2 or not any(
            isinstance(module, VariableSchemaBatchMixin)
            for module in processor.modules()
        ):
            return
        raise ValueError(
            f"Ensemble Recipe {role} with variable-schema processors expects "
            "one logical table with shape [R, C]; additional leading input "
            "dimensions are not supported."
        )

    def _build_class_plan(
        self,
        raw_target: TableTensor,
        transformed_target: EnsembleTable,
        planner: EnsemblePlanner | None,
    ) -> tuple[tuple[object, ...], tuple[torch.Tensor, ...]]:
        local_values = tuple(
            tuple(
                transformed_target[member].categorical.categories[0].tolist()
            )
            for member in range(transformed_target.num_members)
        )
        planned_classes = (
            planner.canonical_classes() if planner is not None else None
        )
        if planned_classes is None:
            raw_values = tuple(raw_target.categorical.categories[0].tolist())
            canonical = tuple(
                value for value in raw_values if value in local_values[0]
            )
            if len(canonical) != len(local_values[0]):
                canonical = local_values[0]
        else:
            canonical = planned_classes

        indices: list[torch.Tensor] = []
        for member, values in enumerate(local_values):
            if len(values) != len(canonical) or any(
                value not in values for value in canonical
            ):
                raise ValueError(
                    "Expected every classification member to contain the "
                    "same fitted classes."
                )
            indices.append(
                torch.tensor(
                    [values.index(value) for value in canonical],
                    dtype=torch.long,
                    device=transformed_target[member].device,
                )
            )
        return canonical, tuple(indices)

    def fit_transform(
        self,
        features: TableTensor,
        target: TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        num_members: int,
        generator: torch.Generator | None = None,
    ) -> tuple[
        EnsembleTable,
        EnsembleTable,
        EnsembleRelatedTables | None,
    ]:
        r"""Fit all Recipe paths and transform context data.

        Args:
            features: Context feature table. Variable-schema paths require
                shape ``[R, C]``.
            target: Context target table. Variable-schema paths require
                shape ``[R, 1]``.
            related_tables: Optional related context tables.
            num_members: Positive number of ensemble members.
            generator: Optional root generator for fit-time randomness.
        """
        if num_members < 1:
            raise ValueError("'num_members' needs to be positive.")
        self._validate_ensemble_composition()
        self._validate_variable_schema_input(
            role="features",
            table=features,
            processor=self.features,
        )
        self._validate_variable_schema_input(
            role="target",
            table=target,
            processor=self._target_processor(self.target),
        )
        if related_tables is not None:
            for table_name, table in related_tables.tables.items():
                self._validate_variable_schema_input(
                    role=f"related table {table_name!r}",
                    table=table,
                    processor=self.features,
                )

        feature_processor = as_ensemble_processor(copy.deepcopy(self.features))
        target_processor = as_ensemble_processor(
            copy.deepcopy(self._target_processor(self.target))
        )
        output_processor = copy.deepcopy(self.output)

        planner = copy.deepcopy(self.ensemble_planner)
        feature_context = EnsembleFitContext.create(
            num_members=num_members,
            table_scope="features",
            generator=generator,
            planner=planner,
        )
        if planner is not None:
            planner.initialize(
                features=features,
                target=target,
                num_members=num_members,
                seed=feature_context.base_seed,
            )
        target_context = EnsembleFitContext.create(
            num_members=num_members,
            table_scope="target",
            base_seed=feature_context.base_seed,
            planner=planner,
        )
        transformed_features = feature_processor.fit_transform_ensemble(
            EnsembleTable.from_shared(
                features,
                num_members=num_members,
            ),
            context=feature_context,
        )
        transformed_target = target_processor.fit_transform_ensemble(
            EnsembleTable.from_shared(
                target,
                num_members=num_members,
            ),
            context=target_context,
        )

        tasks = {
            self._task_from_target(transformed_target[member])
            for member in range(num_members)
        }
        if len(tasks) != 1:
            raise ValueError(
                "All ensemble members must resolve to the same task type."
            )
        task = next(iter(tasks))
        for module in output_processor.modules():
            if isinstance(module, TaskDispatch):
                module._resolve(transformed_target[0])

        related_processors = torch.nn.ModuleList()
        related_table_names: list[str] = []
        transformed_related: EnsembleRelatedTables | None = None
        if related_tables is not None:
            related_outputs: dict[str, EnsembleTable] = {}
            for table_name, table in related_tables.tables.items():
                processor = as_ensemble_processor(copy.deepcopy(self.features))
                related_processors.append(processor)
                related_table_names.append(table_name)
                context = EnsembleFitContext.create(
                    num_members=num_members,
                    table_scope=f"related:{table_name}",
                    base_seed=feature_context.base_seed,
                    planner=planner,
                )
                related_outputs[table_name] = processor.fit_transform_ensemble(
                    EnsembleTable.from_shared(
                        table,
                        num_members=num_members,
                    ),
                    context=context,
                )
            transformed_related = EnsembleRelatedTables(
                tables=related_outputs,
                relationships=related_tables.relationships,
                task_links=related_tables.task_links,
            )

        canonical_classes: tuple[object, ...] | None = None
        class_indices: tuple[torch.Tensor, ...] = ()
        if task == "classification":
            canonical_classes, class_indices = self._build_class_plan(
                target,
                transformed_target,
                planner,
            )

        for module in output_processor.modules():
            if isinstance(module, TargetDecode):
                module._bind(
                    target=target_processor,
                    canonical_classes=canonical_classes,
                    class_indices=class_indices,
                    num_members=num_members,
                )

        # Install the complete fitted graph only after every table succeeds.
        self._ensemble_features = feature_processor
        self._ensemble_related = related_processors
        self._related_table_names = tuple(related_table_names)
        self._ensemble_output = output_processor
        self._num_members = num_members
        return transformed_features, transformed_target, transformed_related

    def transform(
        self,
        features: TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> tuple[EnsembleTable, EnsembleRelatedTables | None]:
        r"""Transform query tables with the fitted ensemble plan.

        Args:
            features: Query feature table. Variable-schema paths require
                shape ``[R, C]``.
            related_tables: Optional related query tables.
        """
        if self._ensemble_features is None:
            raise RuntimeError(
                "'Recipe' is not fitted; call 'fit_transform()' before."
            )
        self._validate_variable_schema_input(
            role="features",
            table=features,
            processor=self.features,
        )
        transformed_features = self._ensemble_features.transform_ensemble(
            EnsembleTable.from_shared(
                features,
                num_members=self._num_members,
            )
        )

        transformed_related: EnsembleRelatedTables | None = None
        if related_tables is not None:
            for table_name, table in related_tables.tables.items():
                self._validate_variable_schema_input(
                    role=f"related table {table_name!r}",
                    table=table,
                    processor=self.features,
                )
            fitted_related = dict(
                zip(self._related_table_names, self._ensemble_related)
            )
            tables = {
                name: cast(
                    EnsembleProcessor,
                    fitted_related[name],
                ).transform_ensemble(
                    EnsembleTable.from_shared(
                        table,
                        num_members=self._num_members,
                    )
                )
                for name, table in related_tables.tables.items()
            }
            transformed_related = EnsembleRelatedTables(
                tables=tables,
                relationships=related_tables.relationships,
                task_links=related_tables.task_links,
            )
        return transformed_features, transformed_related

    def transform_output(
        self,
        outputs: Sequence[TableTensor],
    ) -> TableTensor:
        r"""Run the fitted output pipeline on member-aligned outputs.

        Args:
            outputs: One model output table per stable member.
        """
        if self._ensemble_output is None:
            raise RuntimeError(
                "Recipe is not fitted; call fit_transform() first."
            )
        outputs = tuple(outputs)
        if len(outputs) != self._num_members:
            raise ValueError(
                "Expected one model output per fitted ensemble member."
            )
        return self._ensemble_output.transform(_stack_physical(outputs))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  features={self.features.__repr__(indent=2)[2:]},\n"
            f"  target={self.target.__repr__(indent=2)[2:]},\n"
            f"  output={self.output.__repr__(indent=2)[2:]},\n"
            ")"
        )
