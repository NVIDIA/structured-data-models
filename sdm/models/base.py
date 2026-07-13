import contextlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.processing import InvertibleMixin, Recipe


@dataclass(frozen=True)
class _OutputMapping:
    """Metadata required to map one member output to a common space."""

    is_regression: bool
    member_columns: tuple[str, ...] | None = None
    canonical_columns: tuple[str, ...] | None = None


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


class Model(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`Model` defines the public inferface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    supports_related_tables: ClassVar[bool]

    def __init__(self) -> None:
        super().__init__()

        # One cache per ensemble member.
        self._caches: list[Cache] | None = None
        self._recipes: list[Recipe | None] | None = None

    @_maybe_inference_mode()
    def forward(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
    ) -> Tensor:  # [..., R - R_train, *]
        r"""The in-context learning forward pass.

        Args:
            x: The feature tensor with shape ``[..., R, C]`` with ``R`` rows
                and ``C`` columns.
                The first ``R_train`` rows along ``R`` refer to the in-context
                examples.
            y: The targets of in-context examples with shape
                ``[..., R_train]`` or ``[..., R_train, 1]``.
            related_tables: Additional related context provided to the model.
            recipe: The recipe for pre- and post-processing. If ``None``, no
                recipe is applied.
            num_estimators: The number of estimators for ensembling.

        Returns:
            The prediction for the remaining ``[..., R - R_train]`` test rows.
        """
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        recipes = self._create_member_recipes(
            recipe=recipe,
            num_estimators=num_estimators,
        )
        outs: list[Tensor] = []
        for member_recipe in recipes:
            x_i, y_i, mapping = self._preprocess(
                x,
                y,
                related_tables,
                recipe=member_recipe,
            )
            out = self._forward(x_i, y_i, related_tables, cache=None)
            outs.append(
                self._map_model_output(out, member_recipe, mapping=mapping)
            )

        out = torch.stack(outs).mean(dim=0)
        return self._transform_output(out, recipes[0])

    @torch.inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R_train, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
    ) -> None:
        r"""Fit and cache in-context examples.

        Repeated calls to :meth:`predict` can then reuse the same in-context
        examples while only providing new test rows.

        Args:
            x: The feature tensor with shape ``[..., R_train, C]`` with
                ``R_train`` rows and ``C`` columns.
            y: The targets of in-context examples with shape
                ``[..., R_train]`` or ``[..., R_train, 1]``.
            related_tables: Additional related context provided to the model.
            recipe: The recipe for pre- and post-processing. If ``None``, no
                recipe is applied.
            num_estimators: The number of estimators for ensembling.
        """
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        self.clear()
        recipes = self._create_member_recipes(
            recipe=recipe,
            num_estimators=num_estimators,
        )
        caches: list[Cache] = []
        for member_recipe in recipes:
            x_i, y_i, mapping = self._preprocess(
                x,
                y,
                related_tables,
                recipe=member_recipe,
            )
            x_i = x_i[..., : y_i.size(-1), :]

            cache = Cache(
                {
                    "y.dtype": y_i.dtype,
                    "output.mapping": mapping,
                }
            )
            self._forward(x_i, y_i, related_tables, cache)
            cache.freeze()
            caches.append(cache)

        self._caches = caches
        self._recipes = recipes

    def clear(self) -> None:
        r"""Clear cached in-context examples and fitted member recipes."""
        self._caches = None
        self._recipes = None

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,  # [..., R_test, C]
        related_tables: RelatedTables | None = None,
    ) -> Tensor:  # [..., R_test, *]
        r"""Predict unseen test examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor with shape ``[..., R_test, C]`` with
                ``R_test`` rows and ``C`` columns.
            related_tables: Additional related context provided to the model.

        Returns:
            The prediction for ``[..., R_test]`` test rows.
        """
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        if self._caches is None or self._recipes is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' not yet fitted. Make sure to "
                f"'{self.__class__.__name__}.fit()' beforehand."
            )

        outs: list[Tensor] = []
        for cache, member_recipe in zip(
            self._caches,
            self._recipes,
            strict=True,
        ):
            y_i = torch.empty(
                (*x.size()[:-2], 0),
                dtype=cast(torch.dtype, cache["y.dtype"]),
                device=x.device,
            )
            x_i, y_i, _ = self._preprocess(
                x,
                y_i,
                related_tables,
                recipe=member_recipe,
                fit_recipe=False,
            )
            out = self._forward(x_i, y_i, related_tables, cache)
            mapping = cast(_OutputMapping, cache["output.mapping"])
            outs.append(
                self._map_model_output(
                    out,
                    member_recipe,
                    mapping=mapping,
                )
            )

        out = torch.stack(outs).mean(dim=0)
        return self._transform_output(out, self._recipes[0])

    # Helpers #################################################################

    # FIXME: Remove this guard once variable-length tensor metadata supports
    # every inference-mode view and host-conversion operation used by recipes.
    @torch.inference_mode(False)
    def _preprocess(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None,
        *,
        recipe: Recipe | None = None,
        fit_recipe: bool = True,
    ) -> tuple[Tensor, Tensor, _OutputMapping]:
        if related_tables is not None:
            # TODO Support preprocessing related tables.
            related_tables = None

        canonical_columns = self._classification_columns(y)
        if recipe is not None:
            if not isinstance(x, TableTensor):
                raise ValueError(
                    f"Expected 'x' to be a 'TableTensor' when 'recipe' is "
                    f"given (got '{type(x).__name__}')"
                )
            if fit_recipe:
                if not isinstance(y, TableTensor):
                    raise ValueError(
                        f"Expected 'y' to be a 'TableTensor' when "
                        f"'recipe' is given (got '{type(y).__name__}')"
                    )
                # Fit on the in-context rows only to avoid leakage:
                recipe.features.fit(x[..., : y.size(-2), :])
                y = recipe.target.fit_transform(y)

            x = recipe.features.transform(x)

        member_columns = self._classification_columns(y)
        if canonical_columns is not None and member_columns is not None:
            active = frozenset(member_columns)
            canonical_columns = tuple(
                column for column in canonical_columns if column in active
            )

        if isinstance(x, TableTensor):
            invalid_columns = x.size(-1) - x.numerical.size(-1) - x.id.size(-1)
            if invalid_columns > 0:
                invalid_stypes = [
                    stype.value
                    for stype, tensor in x.items()
                    if tensor.size(-1) > 0
                    and stype not in (Stype.numerical, Stype.id)
                ]
                warnings.warn(
                    f"Expected 'x' to only hold numerical columns but also "
                    f"found {'/'.join(invalid_stypes)} data. "
                    f"This data will be ignored. "
                    f"Make sure that your recipe converts such types to "
                    f"numerical data to include them as features.",
                    stacklevel=2,
                )
            x = x.numerical

        if isinstance(y, TableTensor):
            if y.size(-1) != 1:
                raise ValueError(
                    f"Expected 'y' to refer to a single column "
                    f"(got {y.size(-1)} columns)"
                )

            if y.categorical.numel() > 0:
                y = y.categorical
                if isinstance(y, CategoricalTensor):
                    y = y.as_tensor()
            else:
                assert y.numerical.numel() > 0
                y = y.numerical

        if x.dim() == y.dim() and y.size(-1) == 1:
            y = y.squeeze(-1)

        if x.size()[:-2] != y.size()[:-1]:
            raise ValueError(
                f"Expected 'x' and 'y' to share the same batch dimensions "
                f"(got {tuple(x.size()[:-2])} and {tuple(y.size()[:-1])}"
            )

        mapping = _OutputMapping(
            is_regression=y.is_floating_point(),
            member_columns=member_columns,
            canonical_columns=canonical_columns,
        )
        return x, y, mapping

    def _map_model_output(
        self,
        out: Tensor,  # [..., R_test, *]
        recipe: Recipe | None,
        *,
        mapping: _OutputMapping,
    ) -> Tensor:  # [..., R_test, *]
        if not mapping.is_regression:
            if mapping.member_columns is None:
                return out
            n_classes = len(mapping.member_columns)
            if out.size(-1) < n_classes:
                raise ValueError(
                    "Expected the classification output to contain at least "
                    f"{n_classes} columns (got {out.size(-1)})."
                )
            table = TableTensor.from_tensor(
                out[..., :n_classes].clone(),
                columns=mapping.member_columns,
            )
            if mapping.canonical_columns is not None:
                indices = torch.tensor(
                    [
                        mapping.member_columns.index(column)
                        for column in mapping.canonical_columns
                    ],
                    device=out.device,
                )
                return table.numerical.index_select(-1, indices)
            return table.numerical

        if recipe is None:
            return out

        if not isinstance(recipe.target, InvertibleMixin):
            raise ValueError(
                f"Expected the target steps of 'recipe' to support "
                f"'inverse_transform' to map predictions back to the "
                f"original target space "
                f"(got '{recipe.target.__class__.__name__}')"
            )

        table = TableTensor.from_tensor(out.clone())
        table = recipe.target.inverse_transform(table)

        return table.numerical

    def _transform_output(
        self,
        out: Tensor,  # [..., R_test, *]
        recipe: Recipe | None,
    ) -> Tensor:  # [..., R_test, *]
        table = TableTensor.from_tensor(out.clone())
        if recipe is None:
            return table.numerical
        return recipe.output.transform(table).numerical

    @staticmethod
    def _create_member_recipes(
        recipe: Recipe | None,
        num_estimators: int,
    ) -> list[Recipe | None]:
        if num_estimators <= 0:
            raise ValueError("num_estimators must be positive")
        return [
            None if recipe is None else deepcopy(recipe)
            for _ in range(num_estimators)
        ]

    @staticmethod
    def _classification_columns(
        target: Tensor | TableTensor,
    ) -> tuple[str, ...] | None:
        if not isinstance(target, TableTensor):
            return None
        if target.categorical.size(-1) != 1:
            return None

        category = target.categorical.categories[0]
        # StringTensor host conversion is intentionally outside inference mode;
        # its variable-length storage cannot dispatch the inference-only
        # ``aten.to`` overload used by ``tolist``.
        with torch.inference_mode(False):
            values = category.tolist()
        columns = tuple(str(value) for value in values)
        if len(columns) != len(set(columns)):
            raise ValueError(
                "Expected categorical target values to have unique string "
                "representations for model-output columns."
            )
        return columns

    # Abstract Methods ########################################################

    @abstractmethod
    def _forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        related_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> Tensor:  # [..., R - R_train, *]
        pass

    @classmethod
    @abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""
