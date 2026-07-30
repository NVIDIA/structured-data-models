import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.processing import (
    EnsembleRelatedTables,
    EnsembleTable,
    Recipe,
)
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema


@contextlib.contextmanager
def _maybe_inference_mode() -> Iterator[None]:
    # `torch.inference_mode` is not supported inside a compiled region, so do
    # not enter it when this function is already being compiled.
    # https://github.com/pytorch/pytorch/issues/180823
    # FIXME: Come up with a solution to use torch.compile under
    # torch.inference_mode and remove this workaround.
    if torch.compiler.is_compiling():
        context_fn = contextlib.nullcontext
    else:
        context_fn = torch.inference_mode

    with context_fn():
        yield


class ICLModel(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`ICLModel` defines the public interface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    #: Semantic types supported for input feature columns in this model.
    supported_feature_stypes: ClassVar[frozenset[Stype]]

    #: Semantic types supported for target columns in this model.
    supported_target_stypes: ClassVar[frozenset[Stype]]

    #: Whether this model supports additional related context.
    supports_related_tables: ClassVar[bool]

    #: Whether the model core accepts a leading ensemble dimension.
    supports_vectorized_ensemble: ClassVar[bool] = False

    #: Whether grouped execution preserves member-local generator semantics.
    #:
    #: Leave this disabled when ``_forward`` consumes ``generator`` in a way
    #: whose draws depend on the leading ensemble shape. Such models use the
    #: singleton schedule whenever a generator is supplied.
    supports_vectorized_ensemble_rng: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()

        self._caches: list[Cache] | None = None

    @staticmethod
    def _validate_ensemble_mode(
        ensemble_mode: str,
    ) -> Literal["auto", "parallel", "sequential"]:
        if ensemble_mode not in {"auto", "parallel", "sequential"}:
            raise ValueError(
                "'ensemble_mode' must be 'auto', 'parallel', or 'sequential'."
            )
        return cast(
            Literal["auto", "parallel", "sequential"],
            ensemble_mode,
        )

    @staticmethod
    def _table_execution_signature(table: TableTensor) -> tuple[object, ...]:
        return (
            tuple(table.size()[:-1]),
            tuple(
                (
                    stype,
                    type(block),
                    block.dtype,
                    block.size(-1),
                )
                for stype, block in table.items()
            ),
            tuple(
                category.numel() for category in table.categorical.categories
            ),
        )

    @classmethod
    def _related_execution_signature(
        cls,
        tables: EnsembleRelatedTables | None,
        member: int,
    ) -> tuple[object, ...] | None:
        if tables is None:
            return None
        return tuple(
            (
                name,
                cls._table_execution_signature(table[member]),
            )
            for name, table in sorted(tables.tables.items())
        )

    @classmethod
    def _execution_groups(
        cls,
        *,
        ensemble_mode: Literal["parallel", "sequential"],
        x_context: EnsembleTable,
        y_context: EnsembleTable,
        x_query: EnsembleTable | None,
        related_context_tables: EnsembleRelatedTables | None,
        related_query_tables: EnsembleRelatedTables | None,
    ) -> tuple[tuple[int, ...], ...]:
        if (
            ensemble_mode == "sequential"
            or not cls.supports_vectorized_ensemble
        ):
            return tuple((member,) for member in range(x_context.num_members))

        groups: dict[tuple[object, ...], list[int]] = {}
        for member in range(x_context.num_members):
            signature = (
                cls._table_execution_signature(x_context[member]),
                cls._table_execution_signature(y_context[member]),
                cls._table_execution_signature(x_query[member])
                if x_query is not None
                else None,
                cls._related_execution_signature(
                    related_context_tables,
                    member,
                ),
                cls._related_execution_signature(
                    related_query_tables,
                    member,
                ),
            )
            groups.setdefault(signature, []).append(member)
        return tuple(tuple(members) for members in groups.values())

    @classmethod
    def _materialize_table(
        cls,
        table: EnsembleTable,
        members: tuple[int, ...],
    ) -> TableTensor:
        if cls.supports_vectorized_ensemble:
            return table.materialize(members)
        return table[members[0]]

    @classmethod
    def _materialize_related(
        cls,
        tables: EnsembleRelatedTables | None,
        members: tuple[int, ...],
    ) -> RelatedTables | None:
        if tables is None:
            return None
        if cls.supports_vectorized_ensemble:
            return tables.materialize(members)
        return tables.member(members[0])

    def _validate_ensemble_inputs(
        self,
        *,
        x_context: EnsembleTable,
        y_context: EnsembleTable,
        x_query: EnsembleTable | None,
        related_context_tables: EnsembleRelatedTables | None,
        related_query_tables: EnsembleRelatedTables | None,
    ) -> None:
        for member in range(x_context.num_members):
            context_related = (
                related_context_tables.member(member)
                if related_context_tables is not None
                else None
            )
            self._validate_context(
                x=x_context[member],
                y=y_context[member],
                related_tables=context_related,
            )
            if x_query is None:
                continue
            query_related = (
                related_query_tables.member(member)
                if related_query_tables is not None
                else None
            )
            self._validate_query(
                x_context=x_context[member].schema,
                x_query=x_query[member],
                related_context_tables=(
                    context_related.schema
                    if context_related is not None
                    else None
                ),
                related_query_tables=query_related,
            )

    def _forward_ensemble(
        self,
        *,
        x_context: EnsembleTable,
        y_context: EnsembleTable,
        x_query: EnsembleTable,
        related_context_tables: EnsembleRelatedTables | None,
        related_query_tables: EnsembleRelatedTables | None,
        ensemble_mode: Literal["parallel", "sequential"],
        generator: torch.Generator | None,
        kwargs: dict[str, Any],
    ) -> tuple[TableTensor, ...]:
        groups = self._execution_groups(
            ensemble_mode=ensemble_mode,
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
        )
        if generator is not None and not self.supports_vectorized_ensemble_rng:
            groups = tuple((member,) for group in groups for member in group)
        outputs: list[TableTensor | None] = [None] * x_context.num_members
        for members in groups:
            out = self._forward(
                x_context=self._materialize_table(x_context, members),
                y_context=self._materialize_table(y_context, members),
                x_query=self._materialize_table(x_query, members),
                related_context_tables=self._materialize_related(
                    related_context_tables,
                    members,
                ),
                related_query_tables=self._materialize_related(
                    related_query_tables,
                    members,
                ),
                cache=None,
                generator=generator,
                **kwargs,
            )
            if not self.supports_vectorized_ensemble:
                outputs[members[0]] = out
                continue
            if out.size(0) != len(members):
                raise RuntimeError(
                    "Model output did not preserve the leading ensemble "
                    "dimension."
                )
            for local, member in enumerate(members):
                outputs[member] = out[local]
        return tuple(cast(TableTensor, output) for output in outputs)

    def _fit_ensemble_caches(
        self,
        *,
        recipe: Recipe,
        x_context: EnsembleTable,
        y_context: EnsembleTable,
        related_context_tables: EnsembleRelatedTables | None,
        ensemble_mode: Literal["parallel", "sequential"],
        generator: torch.Generator | None,
        kwargs: dict[str, Any],
    ) -> list[Cache]:
        groups = self._execution_groups(
            ensemble_mode=ensemble_mode,
            x_context=x_context,
            y_context=y_context,
            x_query=None,
            related_context_tables=related_context_tables,
            related_query_tables=None,
        )
        if generator is not None and not self.supports_vectorized_ensemble_rng:
            groups = tuple((member,) for group in groups for member in group)
        caches: list[Cache] = []
        for members in groups:
            x_batch = self._materialize_table(x_context, members)
            y_batch = self._materialize_table(y_context, members)
            related_batch = self._materialize_related(
                related_context_tables,
                members,
            )
            cache = Cache(
                recipe=recipe,
                member_ids=members,
                x_schemas=tuple(
                    x_context[member].schema for member in members
                ),
                related_tables_schemas=tuple(
                    (
                        related_context_tables.member(member).schema
                        if related_context_tables is not None
                        else None
                    )
                    for member in members
                ),
                related_table_names=(
                    tuple(related_context_tables.tables)
                    if related_context_tables is not None
                    else None
                ),
                classes=(
                    y_batch.categorical.categories[0]
                    if y_batch.categorical.size(-1) > 0
                    else None
                ),
                kwargs=kwargs,
            )
            self._forward(
                x_context=x_batch,
                y_context=y_batch,
                x_query=None,
                related_context_tables=related_batch,
                related_query_tables=None,
                cache=cache,
                generator=generator,
                **kwargs,
            )
            if x_context.num_members > 1:
                cache = cache.cpu()
            caches.append(cache.freeze())
        return caches

    @_maybe_inference_mode()
    def forward(
        self,
        x_context: Tensor | TableTensor,  # [..., R_context, D]
        y_context: Tensor | TableTensor,  # [..., R_context, 1]
        x_query: Tensor | TableTensor,  # [..., R_query, D]
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        ensemble_mode: Literal["auto", "parallel", "sequential"] = "auto",
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> TableTensor:  # Recipe-defined output shape.
        r"""The in-context learning forward pass.

        Args:
            x_context: The feature tensor of in-context examples with shape
                ``[..., R_context, D]`` with ``R_context`` rows and ``D``
                columns.
            y_context: The targets of in-context examples with shape
                ``[..., R_context, 1]``.
            x_query: The feature tensor of query examples with shape
                ``[..., R_query, D]`` with ``R_query`` rows and ``D`` columns.
            related_context_tables: Related context for in-context examples.
            related_query_tables: Related context for query examples.
            recipe: The recipe for pre- and post-processing. If ``None``, the
                model's default recipe is applied.
            num_estimators: The number of estimators ``E`` for ensembling.
            ensemble_mode: Whether compatible members execute in parallel,
                sequentially, or in parallel with automatic CUDA out-of-memory
                fallback.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R_query, *]``.
        """
        ensemble_mode = self._validate_ensemble_mode(ensemble_mode)
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(y_context, TableTensor):
            y_context = TableTensor.from_tensor(y_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)

        if (related_context_tables is None) != (related_query_tables is None):
            raise ValueError(
                "Expected 'related_context_tables' and 'related_query_tables' "
                "to be provided together"
            )

        if related_query_tables is not None:
            assert related_context_tables is not None
            related_query_tables = related_query_tables.select_tables(
                tables=related_context_tables.tables
            )

        template = self.default_recipe() if recipe is None else recipe
        fitted_recipe = copy.deepcopy(template)
        x_context_ensemble, y_context_ensemble, related_context_ensemble = (
            fitted_recipe.fit_transform(
                x_context,
                y_context,
                related_context_tables,
                num_members=num_estimators,
                generator=generator,
            )
        )
        x_query_ensemble, related_query_ensemble = fitted_recipe.transform(
            x_query,
            related_query_tables,
        )
        self._validate_ensemble_inputs(
            x_context=x_context_ensemble,
            y_context=y_context_ensemble,
            x_query=x_query_ensemble,
            related_context_tables=related_context_ensemble,
            related_query_tables=related_query_ensemble,
        )

        execution_mode = (
            "parallel" if ensemble_mode == "auto" else ensemble_mode
        )
        generator_state = (
            generator.get_state()
            if ensemble_mode == "auto" and generator is not None
            else None
        )
        try:
            outputs = self._forward_ensemble(
                x_context=x_context_ensemble,
                y_context=y_context_ensemble,
                x_query=x_query_ensemble,
                related_context_tables=related_context_ensemble,
                related_query_tables=related_query_ensemble,
                ensemble_mode=execution_mode,
                generator=generator,
                kwargs=kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            if ensemble_mode != "auto":
                raise
            if generator_state is not None:
                assert generator is not None
                generator.set_state(generator_state)
            torch.cuda.empty_cache()
            outputs = self._forward_ensemble(
                x_context=x_context_ensemble,
                y_context=y_context_ensemble,
                x_query=x_query_ensemble,
                related_context_tables=related_context_ensemble,
                related_query_tables=related_query_ensemble,
                ensemble_mode="sequential",
                generator=generator,
                kwargs=kwargs,
            )

        outputs = tuple(
            cast(TableTensor, output.to(x_query.dtype)) for output in outputs
        )
        return fitted_recipe.transform_output(outputs)

    @_maybe_inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        y: Tensor | TableTensor,  # [..., R, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        ensemble_mode: Literal["auto", "parallel", "sequential"] = "auto",
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Fit and cache in-context examples.

        Repeated calls to :meth:`predict` can then reuse the same in-context
        examples while only providing new query examples.

        Args:
            x: The feature tensor of in-context examples with shape
                ``[..., R, D]`` with ``R`` rows and ``C`` columns.
            y: The targets of in-context examples with shape
                ``[..., R, 1]``.
            related_tables: Related context for in-context examples.
            recipe: The recipe for pre- and post-processing. If ``None``, the
                model's default recipe is applied.
            num_estimators: The number of estimators for ensembling.
            ensemble_mode: Whether compatible members execute in parallel,
                sequentially, or in parallel with automatic CUDA out-of-memory
                fallback.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.
        """
        ensemble_mode = self._validate_ensemble_mode(ensemble_mode)
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if not isinstance(y, TableTensor):
            y = TableTensor.from_tensor(y)

        template = self.default_recipe() if recipe is None else recipe
        fitted_recipe = copy.deepcopy(template)
        x_ensemble, y_ensemble, related_ensemble = fitted_recipe.fit_transform(
            x,
            y,
            related_tables,
            num_members=num_estimators,
            generator=generator,
        )
        self._validate_ensemble_inputs(
            x_context=x_ensemble,
            y_context=y_ensemble,
            x_query=None,
            related_context_tables=related_ensemble,
            related_query_tables=None,
        )

        execution_mode = (
            "parallel" if ensemble_mode == "auto" else ensemble_mode
        )
        generator_state = (
            generator.get_state()
            if ensemble_mode == "auto" and generator is not None
            else None
        )
        try:
            caches = self._fit_ensemble_caches(
                recipe=fitted_recipe,
                x_context=x_ensemble,
                y_context=y_ensemble,
                related_context_tables=related_ensemble,
                ensemble_mode=execution_mode,
                generator=generator,
                kwargs=kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            if ensemble_mode != "auto":
                raise
            if generator_state is not None:
                assert generator is not None
                generator.set_state(generator_state)
            torch.cuda.empty_cache()
            caches = self._fit_ensemble_caches(
                recipe=fitted_recipe,
                x_context=x_ensemble,
                y_context=y_ensemble,
                related_context_tables=related_ensemble,
                ensemble_mode="sequential",
                generator=generator,
                kwargs=kwargs,
            )
        self._caches = caches

    @_maybe_inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        r"""Predict unseen query examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor of query examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            related_tables: Related context for query examples.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R, *]``.
        """
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)

        if self._caches is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} not yet fitted. Make sure to "
                f"call '{self.__class__.__name__}.fit()' before."
            )

        first_cache = self._caches[0]
        recipe = cast(Recipe, first_cache["recipe"])
        fitted_related_names = cast(
            tuple[str, ...] | None,
            first_cache["related_table_names"],
        )
        if (related_tables is None) != (fitted_related_names is None):
            raise ValueError("Expected related tables to be provided together")
        if related_tables is not None:
            assert fitted_related_names is not None
            if len(fitted_related_names) == 0:
                raise ValueError(
                    "Expected related tables to be provided together"
                )
            related_tables = related_tables.select_tables(
                tables=fitted_related_names,
            )

        x_ensemble, related_ensemble = recipe.transform(x, related_tables)
        outputs: list[TableTensor | None] = [None] * x_ensemble.num_members
        for cache in self._caches:
            members = cast(tuple[int, ...], cache["member_ids"])
            x_schemas = cast(tuple[TableSchema, ...], cache["x_schemas"])
            related_schemas = cast(
                tuple[RelatedTablesSchema | None, ...],
                cache["related_tables_schemas"],
            )
            for local, member in enumerate(members):
                self._validate_query(
                    x_context=x_schemas[local],
                    x_query=x_ensemble[member],
                    related_context_tables=related_schemas[local],
                    related_query_tables=(
                        related_ensemble.member(member)
                        if related_ensemble is not None
                        else None
                    ),
                )

            out = self._forward(
                x_context=None,
                y_context=None,
                x_query=self._materialize_table(x_ensemble, members),
                related_context_tables=None,
                related_query_tables=self._materialize_related(
                    related_ensemble,
                    members,
                ),
                cache=cache.to(x_ensemble.device),
                generator=None,
                **cast(dict[str, Any], cache["kwargs"]),
            )
            if not self.supports_vectorized_ensemble:
                outputs[members[0]] = cast(
                    TableTensor,
                    out.to(x.dtype),
                )
                continue
            if out.size(0) != len(members):
                raise RuntimeError(
                    "Model output did not preserve the leading ensemble "
                    "dimension."
                )
            for local, member in enumerate(members):
                outputs[member] = cast(
                    TableTensor,
                    out[local].to(x.dtype),
                )

        return recipe.transform_output(
            tuple(cast(TableTensor, output) for output in outputs)
        )

    def clear(self) -> None:
        r"""Clear cached context state created by :meth:`fit`."""
        self._caches = None

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        device_repr = f"device={device}" if device.type != "cpu" else ""
        return f"{self.__class__.__name__}({device_repr})"

    # Abstract Methods ########################################################

    @abstractmethod
    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, *]
        pass

    @classmethod
    @abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    # Helpers #################################################################

    def _validate_context(
        self,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables | None,
    ) -> None:

        if y.size(-1) != 1:
            raise ValueError(
                f"Expected target to have exactly one column "
                f"(got {y.size(-1)})"
            )
        if x.size()[:-1] != y.size()[:-1]:
            raise ValueError(
                f"Expected features and targets to have matching row "
                f"dimensions (got {tuple(x.size()[:-1])} and "
                f"{tuple(y.size()[:-1])})"
            )
        invalid = x.active_stypes - self.supported_feature_stypes - {Stype.id}
        if len(invalid) > 0:
            raise ValueError(
                f"{self.__class__.__name__!r} received unsupported feature "
                f"stypes: {', '.join(stype.value for stype in invalid)}"
            )
        invalid = y.active_stypes - self.supported_target_stypes
        if len(invalid) > 0:
            raise ValueError(
                f"{self.__class__.__name__!r} received unsupported target "
                f"stypes: {', '.join(stype.value for stype in invalid)}"
            )

        if related_tables is not None:
            if not self.supports_related_tables:
                raise ValueError(
                    f"{self.__class__.__name__!r} does not support related "
                    f"tables"
                )
            for table_name, table in related_tables.tables.items():
                invalid = table.active_stypes - self.supported_feature_stypes
                invalid = invalid - {Stype.id}
                if len(invalid) > 0:
                    raise ValueError(
                        f"{self.__class__.__name__!r} received unsupported "
                        f"feature stypes in related table {table_name!r}: "
                        f"{', '.join(stype.value for stype in invalid)}"
                    )

    def _validate_query(
        self,
        x_context: TableSchema,
        x_query: TableTensor,
        related_context_tables: RelatedTablesSchema | None,
        related_query_tables: RelatedTables | None,
    ) -> None:

        if x_context != x_query.schema:
            raise ValueError(
                "Expected context and query features to share the same schema"
            )

        if (related_context_tables is None) != (related_query_tables is None):
            raise ValueError("Expected related tables to be provided together")

        if related_context_tables is not None:
            assert related_query_tables is not None
            if not related_query_tables.schema.is_subset_of(
                related_context_tables
            ):
                raise ValueError(
                    "Expected related context and query tables to share the "
                    "same schema"
                )
