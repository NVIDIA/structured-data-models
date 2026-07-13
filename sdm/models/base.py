import contextlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.processing import InvertibleMixin, Recipe


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

    :class:`Model` defines the public interface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    supports_related_tables: ClassVar[bool]

    def __init__(self) -> None:
        super().__init__()

        # One cache per ensemble member.
        self._caches: list[Cache] | None = None
        self._recipe: Recipe | None = None

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

        # Align every member output to the input target's class order before
        # aggregating the ensemble.
        original_class_labels = self._class_labels(y)
        outs: list[Tensor] = []
        for _ in range(num_estimators):
            # TODO Iterate over Recipes instead of using a single recipe once
            # Recipe adds support for multiple recipes.
            x_i, y_i, member_class_labels = self._preprocess(
                x,
                y,
                related_tables,
                recipe=recipe,
            )
            out = self._forward(x_i, y_i, related_tables, cache=None)
            out = self._postprocess(
                out,
                y_i,
                recipe,
                member_class_labels=member_class_labels,
                original_class_labels=original_class_labels,
            )
            outs.append(out)

        out = torch.stack(outs).mean(dim=0)
        table = TableTensor.from_tensor(out.clone())
        if recipe is None:
            return table.numerical
        return recipe.output.transform(table).numerical

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
        original_class_labels = self._class_labels(y)
        caches: list[Cache] = []
        for _ in range(num_estimators):
            # TODO Iterate over Recipes instead of using a single recipe once
            # Recipe adds support for multiple recipes.
            x_i, y_i, member_class_labels = self._preprocess(
                x,
                y,
                related_tables,
                recipe=recipe,
            )
            x_i = x_i[..., : y_i.size(-1), :]

            cache = Cache(
                {
                    "y.dtype": y_i.dtype,
                    # predict() only receives features, so retain the fitted
                    # target order needed to map this member's output.
                    "target.member_class_labels": member_class_labels,
                    "target.original_class_labels": original_class_labels,
                }
            )
            self._forward(x_i, y_i, related_tables, cache)
            cache.freeze()
            caches.append(cache)

        self._caches = caches
        # TODO: Once creating Recipes from a Recipe is supported, we should
        # iterate over the recipes so that every predict call runs a consistent
        # recipe per ensemble member.
        self._recipe = recipe

    def clear(self) -> None:
        r"""Clears cached in-context examples and the fitted recipe."""
        self._caches = None
        self._recipe = None

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

        if self._caches is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' not yet fitted. Make sure to "
                f"'{self.__class__.__name__}.fit()' beforehand."
            )

        recipe = self._recipe
        outs: list[Tensor] = []
        for cache in self._caches:
            y_i = torch.empty(
                (*x.size()[:-2], 0),
                dtype=cast(torch.dtype, cache["y.dtype"]),
                device=x.device,
            )
            # TODO Iterate over Recipes instead of using a single recipe once
            # Recipe adds support for multiple recipes.
            x_i, y_i, _ = self._preprocess(
                x,
                y_i,
                related_tables,
                recipe=recipe,
                fit_recipe=False,
            )
            out = self._forward(x_i, y_i, related_tables, cache)
            member_class_labels = cast(
                tuple[str, ...] | None,
                cache["target.member_class_labels"],
            )
            original_class_labels = cast(
                tuple[str, ...] | None,
                cache["target.original_class_labels"],
            )
            out = self._postprocess(
                out,
                y_i,
                recipe,
                member_class_labels=member_class_labels,
                original_class_labels=original_class_labels,
            )
            outs.append(out)

        out = torch.stack(outs).mean(dim=0)
        table = TableTensor.from_tensor(out.clone())
        if recipe is None:
            return table.numerical
        return recipe.output.transform(table).numerical

    # Helpers #################################################################

    # Recipe transforms can create variable-length category metadata while the
    # public model call runs in inference mode. StringTensor does not yet
    # support the inference-only host conversion used to read that metadata.
    @torch.inference_mode(False)
    def _preprocess(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None,
        *,
        recipe: Recipe | None = None,
        fit_recipe: bool = True,
    ) -> tuple[Tensor, Tensor, tuple[str, ...] | None]:
        if related_tables is not None:
            # TODO Support preprocessing related tables.
            related_tables = None

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

        # A target processor may assign a different class-code order to each
        # ensemble member; the model head follows this transformed order.
        member_class_labels = self._class_labels(y)

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

        return x, y, member_class_labels

    def _postprocess(
        self,
        out: Tensor,  # [..., R_test, *]
        y: Tensor,  # [..., R_train]
        recipe: Recipe | None,
        *,
        member_class_labels: tuple[str, ...] | None,
        original_class_labels: tuple[str, ...] | None,
    ) -> Tensor:  # [..., R_test, *]
        if y.is_floating_point():
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
            return recipe.target.inverse_transform(table).numerical

        # Raw tensor targets have no class-label metadata and therefore no
        # recipe-induced class order to restore.
        if member_class_labels is None:
            return out

        n_classes = len(member_class_labels)
        if out.size(-1) < n_classes:
            raise ValueError(
                "Expected the classification output to contain at least "
                f"{n_classes} columns (got {out.size(-1)})."
            )
        out = out[..., :n_classes]

        assert original_class_labels is not None
        if member_class_labels == original_class_labels:
            return out

        member_indices = {
            label: index for index, label in enumerate(member_class_labels)
        }
        indices = torch.tensor(
            [
                member_indices[label]
                for label in original_class_labels
                if label in member_indices
            ],
            device=out.device,
        )
        return out.index_select(-1, indices)

    @staticmethod
    def _class_labels(
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
        labels = tuple(str(value) for value in values)
        if len(labels) != len(set(labels)):
            raise ValueError(
                "Expected categorical targets to have unique class-label "
                "representations."
            )
        return labels

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
