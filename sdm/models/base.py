import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.explain.base import (
    ExplanationMethod,
    ExplanationMode,
    ExplanationReplacements,
    UnsupportedExplanationError,
)
from sdm.explain.execution import PreparedFittedInputs, PreparedICLInputs
from sdm.explain.result import Explanation, InputSite, OutputIndex
from sdm.processing import InvertibleMixin, Processor, Recipe
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


@contextlib.contextmanager
def _explanation_mode(*, gradients: bool) -> Iterator[None]:
    with torch.inference_mode(False):
        context_fn = torch.enable_grad if gradients else torch.no_grad
        with context_fn():
            yield


@dataclass(frozen=True)
class _FullContextInputs:
    context: TableTensor
    target: TableTensor
    query: TableTensor
    related_context: RelatedTables | None
    related_query: RelatedTables | None


@dataclass(frozen=True)
class _PreparedICLMember:
    inputs: PreparedICLInputs
    target: TableTensor
    recipe: Recipe
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class _ExplanationRandomState:
    generator: torch.Generator | None
    cpu: Tensor
    cuda: Tensor | None
    device: torch.device


@dataclass(frozen=True)
class _FittedQueryInputs:
    query: TableTensor
    related_query: RelatedTables | None


@dataclass(frozen=True)
class _PreparedFittedMember:
    inputs: PreparedFittedInputs
    cache: Cache
    recipe: Recipe
    kwargs: Mapping[str, Any]


class ICLModel(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`ICLModel` defines the public interface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]]
    supported_target_stypes: ClassVar[frozenset[Stype]]
    supports_related_tables: ClassVar[bool]

    def __init__(self) -> None:
        super().__init__()

        # One cache per ensemble member.
        self._caches: list[Cache] | None = None

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
            recipe: The recipe for pre- and post-processing.
            num_estimators: The number of estimators for ensembling.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction. Member outputs enter ``recipe.output``
            stacked as ``[E, ..., R_query, *]``; the output processors
            determine whether the leading estimator dimension remains.
        """
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")
        inputs = self._full_context_inputs(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
        )
        recipe = self.default_recipe() if recipe is None else recipe
        recipes = tuple(copy.deepcopy(recipe) for _ in range(num_estimators))

        outs: list[TableTensor] = []
        for member_recipe in recipes:
            prepared = self._prepare_full_context_member(
                inputs=inputs,
                recipe=member_recipe,
                generator=generator,
                kwargs=kwargs,
            )
            outs.append(
                self._execute_full_context_member(
                    prepared,
                    generator=generator,
                )
            )

        return self._finalize_outputs(
            outs,
            recipe=recipes[-1],
            dtype=prepared.inputs.query.dtype,
        )

    def explain_full_context(
        self,
        method: ExplanationMethod,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        target: OutputIndex,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Explanation:
        r"""Explain a cache-free prediction with explicit context inputs."""
        self._validate_explanation_method(
            method,
            mode=ExplanationMode.full_context,
        )
        if num_estimators != 1:
            raise UnsupportedExplanationError(
                "Full-context explanation currently requires exactly one "
                "estimator"
            )
        if any(module.training for module in self.modules()):
            raise UnsupportedExplanationError(
                "Full-context explanation requires the model to be fully "
                "in evaluation mode"
            )

        with _explanation_mode(gradients=method.requirements.gradients):
            inputs = self._full_context_inputs(
                x_context=x_context,
                y_context=y_context,
                x_query=x_query,
                related_context_tables=related_context_tables,
                related_query_tables=related_query_tables,
            )
            inputs = _normal_full_context_inputs(inputs)
            with _isolated_randomness(
                generator,
                device=inputs.context.device,
            ) as preparation_generator:
                prepared = self._prepare_full_context_member(
                    inputs=inputs,
                    recipe=copy.deepcopy(
                        self.default_recipe() if recipe is None else recipe
                    ),
                    generator=preparation_generator,
                    kwargs=kwargs,
                )
                execution_state = _capture_random_state(
                    preparation_generator,
                    device=inputs.context.device,
                )

                def evaluate(
                    replacements: ExplanationReplacements | None = None,
                ) -> TableTensor:
                    with _replay_randomness(
                        execution_state
                    ) as execution_generator:
                        out = self._execute_full_context_member(
                            prepared,
                            replacements=replacements,
                            generator=execution_generator,
                        )
                        return self._finalize_outputs(
                            (out,),
                            recipe=prepared.recipe,
                            dtype=prepared.inputs.query.dtype,
                        )

                prediction = evaluate(None)
                prediction = prediction.replace_blocks(
                    numerical=prediction.numerical.detach()
                )
                explanation = method.explain(
                    evaluate=evaluate,
                    inputs=prepared.inputs.sites,
                    prediction=prediction,
                    target=target,
                    mode=ExplanationMode.full_context,
                )

        return self._validate_explanation_result(
            explanation,
            method=method,
            mode=ExplanationMode.full_context,
            prediction=prediction,
            inputs=prepared.inputs.sites,
            target=target,
        )

    def _full_context_inputs(
        self,
        *,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
    ) -> _FullContextInputs:
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

        return _FullContextInputs(
            context=x_context,
            target=y_context,
            query=x_query,
            related_context=related_context_tables,
            related_query=related_query_tables,
        )

    def _prepare_full_context_member(
        self,
        *,
        inputs: _FullContextInputs,
        recipe: Recipe,
        generator: torch.Generator | None,
        kwargs: Mapping[str, Any],
    ) -> _PreparedICLMember:
        x_context = recipe.features.fit_transform(
            inputs.context,
            generator=generator,
        )
        y_context = recipe.target.fit_transform(
            inputs.target,
            generator=generator,
        )
        x_query = recipe.features.transform(inputs.query)

        related_context = related_query = None
        if inputs.related_context is not None:
            related_processors = {
                table_name: copy.deepcopy(recipe.features)
                for table_name in inputs.related_context.tables
            }
            related_context = replace(
                inputs.related_context,
                tables={
                    name: related_processors[name].fit_transform(
                        table,
                        generator=generator,
                    )
                    for name, table in inputs.related_context.tables.items()
                },
            )
            assert inputs.related_query is not None
            related_query = replace(
                inputs.related_query,
                tables={
                    name: related_processors[name].transform(table)
                    for name, table in inputs.related_query.tables.items()
                },
            )

        self._validate_context(
            x=x_context,
            y=y_context,
            related_tables=related_context,
        )
        self._validate_query(
            x_context=x_context.schema,
            x_query=x_query,
            related_context_tables=(
                related_context.schema if related_context is not None else None
            ),
            related_query_tables=related_query,
        )
        return _PreparedICLMember(
            inputs=PreparedICLInputs(
                context=x_context,
                query=x_query,
                related_context=related_context,
                related_query=related_query,
            ),
            target=y_context,
            recipe=recipe,
            kwargs=dict(kwargs),
        )

    def _execute_full_context_member(
        self,
        prepared: _PreparedICLMember,
        *,
        replacements: ExplanationReplacements | None = None,
        generator: torch.Generator | None,
    ) -> TableTensor:
        inputs = prepared.inputs.replace_numerical(replacements)
        out = self._forward(
            x_context=inputs.context,
            y_context=prepared.target,
            x_query=inputs.query,
            related_context_tables=inputs.related_context,
            related_query_tables=inputs.related_query,
            cache=None,
            generator=generator,
            **prepared.kwargs,
        )
        if prepared.target.numerical.size(-1) == 1:
            if not isinstance(prepared.recipe.target, InvertibleMixin):
                raise RuntimeError("Target recipe is not invertible")
            out = prepared.recipe.target.inverse_transform(out)
        return out

    def _finalize_outputs(
        self,
        outs: Sequence[TableTensor],
        *,
        recipe: Recipe,
        dtype: torch.dtype,
    ) -> TableTensor:
        out = cast(TableTensor, torch.stack(tuple(outs), dim=0))
        out = cast(TableTensor, out.to(dtype))
        return recipe.output.transform(out)

    def _validate_explanation_method(
        self,
        method: ExplanationMethod,
        *,
        mode: ExplanationMode,
    ) -> None:
        if not isinstance(method, ExplanationMethod):
            raise TypeError("'method' needs to be an 'ExplanationMethod'")
        if mode not in method.requirements.supported_modes:
            raise UnsupportedExplanationError(
                f"'{method.name}' does not support '{mode.value}' execution"
            )
        if method.requirements.input_space != "processed":
            raise UnsupportedExplanationError(
                f"'{self.__class__.__name__}' does not yet expose "
                f"'{method.requirements.input_space}' explanation inputs"
            )
        method.validate_model(self, mode=mode)

    def _validate_explanation_result(
        self,
        explanation: Explanation,
        *,
        method: ExplanationMethod,
        mode: ExplanationMode,
        prediction: TableTensor,
        inputs: Mapping[InputSite, TableTensor],
        target: OutputIndex,
    ) -> Explanation:
        if not isinstance(explanation, Explanation):
            raise TypeError(
                f"'{method.name}.explain()' needs to return an 'Explanation'"
            )
        if explanation.method != method.name:
            raise ValueError(
                "Explanation method identity does not match the requested "
                "method"
            )
        if explanation.mode != mode:
            raise ValueError(
                "Explanation mode does not match the model execution path"
            )
        if explanation.target != target.resolve(prediction):
            raise ValueError(
                "Explanation target does not match the requested output"
            )
        if explanation.prediction.numerical.requires_grad:
            raise ValueError(
                "Explanation prediction needs to be detached from autograd"
            )
        if not explanation.prediction.allclose(
            prediction,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise ValueError(
                "Explanation prediction does not match the unmodified model "
                "prediction"
            )

        for attribution in explanation.attributions:
            if attribution.site not in inputs:
                raise ValueError(
                    "Explanation attribution site was not exposed by the "
                    f"model: {attribution.site!r}"
                )
            if attribution.input_space != method.requirements.input_space:
                raise ValueError(
                    "Explanation attribution input space does not match the "
                    "method requirements"
                )
            if attribution.values.numerical.requires_grad:
                raise ValueError(
                    "Explanation attributions need to be detached from "
                    "autograd"
                )
            _validate_attribution_alignment(
                attribution.values,
                inputs[attribution.site],
            )
        return explanation

    @_maybe_inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        y: Tensor | TableTensor,  # [..., R, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
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
            recipe: The recipe for pre- and post-processing. If ``None``, no
                recipe is applied.
            num_estimators: The number of estimators for ensembling.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.
        """
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if not isinstance(y, TableTensor):
            y = TableTensor.from_tensor(y)

        recipe = self.default_recipe() if recipe is None else recipe
        recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]

        self.clear()
        caches: list[Cache] = []
        for recipe in recipes:
            x_i = recipe.features.fit_transform(x, generator=generator)
            y_i = recipe.target.fit_transform(y, generator=generator)

            related_tables_i = None
            related_processors = None
            if related_tables is not None:
                related_processors = {
                    table_name: copy.deepcopy(recipe.features)
                    for table_name in related_tables.tables
                }
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: related_processors[name].fit_transform(
                            t,
                            generator=generator,
                        )
                        for name, t in related_tables.tables.items()
                    },
                )

            self._validate_context(
                x=x_i,
                y=y_i,
                related_tables=related_tables_i,
            )

            cache = Cache(
                recipe=recipe,
                x_schema=x_i.schema,
                related_processors=related_processors,
                related_tables_schema=related_tables_i.schema
                if related_tables_i is not None
                else None,
                classes=y_i.categorical.categories[0]
                if y_i.categorical.size(-1) > 0
                else None,
                kwargs=kwargs,
            )

            self._forward(
                x_context=x_i,
                y_context=y_i,
                x_query=None,
                related_context_tables=related_tables_i,
                related_query_tables=None,
                cache=cache,
                generator=generator,
                **kwargs,
            )
            cache = cache.cpu().freeze()
            caches.append(cache)
        self._caches = caches

    def clear(self) -> None:
        r"""Clears cached in-context examples and the fitted recipe."""
        self._caches = None

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
            The processed prediction. Member outputs enter ``recipe.output``
            stacked as ``[E, ..., R, *]``; the output processors determine
            whether the leading estimator dimension remains.
        """
        inputs = self._fitted_query_inputs(
            x=x,
            related_tables=related_tables,
        )
        assert self._caches is not None

        outs: list[TableTensor] = []
        for cache in self._caches:
            prepared = self._prepare_fitted_member(
                inputs=inputs,
                cache=cache,
            )
            outs.append(self._execute_fitted_member(prepared))

        return self._finalize_outputs(
            outs,
            recipe=prepared.recipe,
            dtype=prepared.inputs.query.dtype,
        )

    def explain_fitted(
        self,
        method: ExplanationMethod,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
        *,
        target: OutputIndex,
    ) -> Explanation:
        r"""Explain a query prediction using state created by :meth:`fit`."""
        self._validate_explanation_method(
            method,
            mode=ExplanationMode.fitted,
        )
        if method.requirements.gradients:
            raise UnsupportedExplanationError(
                "Fitted explanation does not yet support gradients because "
                "fit caches contain inference tensors"
            )
        if any(module.training for module in self.modules()):
            raise UnsupportedExplanationError(
                "Fitted explanation requires the model to be fully in "
                "evaluation mode"
            )

        with _explanation_mode(gradients=False):
            inputs = self._fitted_query_inputs(
                x=x_query,
                related_tables=related_query_tables,
            )
            inputs = _normal_fitted_query_inputs(inputs)
            assert self._caches is not None
            if len(self._caches) != 1:
                raise UnsupportedExplanationError(
                    "Fitted explanation currently requires exactly one "
                    "estimator"
                )
            with _isolated_randomness(
                None,
                device=inputs.query.device,
            ):
                prepared = self._prepare_fitted_member(
                    inputs=inputs,
                    cache=self._caches[0],
                )
                execution_state = _capture_random_state(
                    None,
                    device=inputs.query.device,
                )

                def evaluate(
                    replacements: ExplanationReplacements | None = None,
                ) -> TableTensor:
                    with (
                        _replay_randomness(execution_state),
                        _explanation_mode(gradients=False),
                    ):
                        out = self._execute_fitted_member(
                            prepared,
                            replacements=replacements,
                        )
                        return self._finalize_outputs(
                            (out,),
                            recipe=prepared.recipe,
                            dtype=prepared.inputs.query.dtype,
                        )

                prediction = evaluate(None)
                explanation = method.explain(
                    evaluate=evaluate,
                    inputs=prepared.inputs.sites,
                    prediction=prediction,
                    target=target,
                    mode=ExplanationMode.fitted,
                )

        return self._validate_explanation_result(
            explanation,
            method=method,
            mode=ExplanationMode.fitted,
            prediction=prediction,
            inputs=prepared.inputs.sites,
            target=target,
        )

    def _fitted_query_inputs(
        self,
        *,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None,
    ) -> _FittedQueryInputs:
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if self._caches is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' not yet fitted. Make sure to "
                f"call '{self.__class__.__name__}.fit()' before."
            )

        if related_tables is not None:
            if self._caches[0]["related_tables_schema"] is None:
                raise ValueError(
                    "Expected related tables to be provided together"
                )
            related_tables = related_tables.select_tables(
                tables=cast(
                    RelatedTablesSchema,
                    self._caches[0]["related_tables_schema"],
                ).tables,
            )
        return _FittedQueryInputs(
            query=x,
            related_query=related_tables,
        )

    def _prepare_fitted_member(
        self,
        *,
        inputs: _FittedQueryInputs,
        cache: Cache,
    ) -> _PreparedFittedMember:
        recipe = cast(Recipe, cache["recipe"])
        query = recipe.features.transform(inputs.query)

        related_query = None
        if inputs.related_query is not None:
            related_processors = cast(
                Mapping[str, Processor],
                cache["related_processors"],
            )
            related_query = replace(
                inputs.related_query,
                tables={
                    name: related_processors[name].transform(table)
                    for name, table in inputs.related_query.tables.items()
                },
            )

        self._validate_query(
            x_context=cast(TableSchema, cache["x_schema"]),
            x_query=query,
            related_context_tables=cast(
                RelatedTablesSchema,
                cache["related_tables_schema"],
            ),
            related_query_tables=related_query,
        )
        return _PreparedFittedMember(
            inputs=PreparedFittedInputs(
                query=query,
                related_query=related_query,
            ),
            cache=cache.to(query.device),
            recipe=recipe,
            kwargs=cast(dict[str, Any], cache["kwargs"]),
        )

    def _execute_fitted_member(
        self,
        prepared: _PreparedFittedMember,
        *,
        replacements: ExplanationReplacements | None = None,
    ) -> TableTensor:
        inputs = prepared.inputs.replace_numerical(replacements)
        out = self._forward(
            x_context=None,
            y_context=None,
            x_query=inputs.query,
            related_context_tables=None,
            related_query_tables=inputs.related_query,
            cache=prepared.cache,
            generator=None,
            **prepared.kwargs,
        )
        if prepared.cache["classes"] is None:
            if not isinstance(prepared.recipe.target, InvertibleMixin):
                raise RuntimeError("Target recipe is not invertible")
            out = prepared.recipe.target.inverse_transform(out)
        return out

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
                f"'{self.__class__.__name__}' received unsupported feature "
                f"stypes: {', '.join(stype.value for stype in invalid)}"
            )
        invalid = y.active_stypes - self.supported_target_stypes
        if len(invalid) > 0:
            raise ValueError(
                f"'{self.__class__.__name__}' received unsupported target "
                f"stypes: {', '.join(stype.value for stype in invalid)}"
            )

        if related_tables is not None:
            if not self.supports_related_tables:
                raise ValueError(
                    f"'{self.__class__.__name__}' does not support related "
                    f"tables"
                )
            for table_name, table in related_tables.tables.items():
                invalid = table.active_stypes - self.supported_feature_stypes
                invalid = invalid - {Stype.id}
                if len(invalid) > 0:
                    raise ValueError(
                        f"'{self.__class__.__name__}' received unsupported "
                        f"feature stypes in related table '{table_name}': "
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


def _normal_full_context_inputs(
    inputs: _FullContextInputs,
) -> _FullContextInputs:
    return _FullContextInputs(
        context=_normal_table(inputs.context),
        target=_normal_table(inputs.target),
        query=_normal_table(inputs.query),
        related_context=_normal_related(inputs.related_context),
        related_query=_normal_related(inputs.related_query),
    )


def _normal_fitted_query_inputs(
    inputs: _FittedQueryInputs,
) -> _FittedQueryInputs:
    return _FittedQueryInputs(
        query=_normal_table(inputs.query),
        related_query=_normal_related(inputs.related_query),
    )


def _normal_table(table: TableTensor) -> TableTensor:
    if table.is_inference() or any(
        block.is_inference() for _, block in table.items()
    ):
        return cast(TableTensor, table.clone())
    return table


def _normal_related(
    related: RelatedTables | None,
) -> RelatedTables | None:
    if related is None:
        return None
    tables = {
        name: _normal_table(table) for name, table in related.tables.items()
    }
    if all(table is related.tables[name] for name, table in tables.items()):
        return related
    return replace(related, tables=tables)


def _validate_attribution_alignment(
    values: TableTensor,
    inputs: TableTensor,
) -> None:
    if values.size()[:-1] != inputs.size()[:-1]:
        raise ValueError(
            "Explanation attribution rows do not match the model input"
        )
    for stype in (Stype.numerical, Stype.id):
        if values.columns[stype] != inputs.columns[stype]:
            raise ValueError(
                "Explanation attribution columns do not match the model input"
            )
    if not values.id.equal(inputs.id):
        raise ValueError(
            "Explanation attribution identifiers do not match the model input"
        )
    if values.device != inputs.device:
        raise ValueError(
            "Explanation attribution device does not match the model input"
        )


def _clone_generator(generator: torch.Generator) -> torch.Generator:
    clone = torch.Generator(device=generator.device)
    clone.set_state(generator.get_state())
    return clone


@contextlib.contextmanager
def _isolated_randomness(
    generator: torch.Generator | None,
    *,
    device: torch.device,
) -> Iterator[torch.Generator | None]:
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield _clone_generator(generator) if generator is not None else None


def _capture_random_state(
    generator: torch.Generator | None,
    *,
    device: torch.device,
) -> _ExplanationRandomState:
    return _ExplanationRandomState(
        generator=(
            _clone_generator(generator) if generator is not None else None
        ),
        cpu=torch.get_rng_state(),
        cuda=(
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
        device=device,
    )


@contextlib.contextmanager
def _replay_randomness(
    state: _ExplanationRandomState,
) -> Iterator[torch.Generator | None]:
    devices = [state.device] if state.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state.cpu)
        if state.cuda is not None:
            torch.cuda.set_rng_state(state.cuda, state.device)
        yield (
            _clone_generator(state.generator)
            if state.generator is not None
            else None
        )
