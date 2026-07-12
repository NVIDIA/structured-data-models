import contextlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar, cast

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


class BaseModel(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`BaseModel` defines the public inferface shared among in-context
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
        seqused_train: Tensor | None = None,  # [...]
        seqused_cols: Tensor | None = None,  # []
        batch_size_limit: int | None = None,
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
            seqused_train: Valid in-context example counts with shape ``[...]``
                and :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                Counts must be positive. When set, only the first
                ``seqused_train`` of the ``R_train`` in-context rows act as
                context, and the remaining rows are treated as padding: they
                are masked from every attention key/value stream and cannot
                influence any prediction, provided the padded feature
                entries are finite and of moderate magnitude (``0`` is
                recommended - masking adds ``-inf`` to attention logits
                after the query/key product, so non-finite or overflowing
                padded values poison the softmax with ``NaN``).
                Padded ``y`` entries must still be valid targets (for example
                ``0``). Together with padded test rows (whose extra outputs
                callers simply discard), this lets streams of varying table
                sizes be padded to a small set of bucketed shapes so compiled
                graphs and per-shape kernel selection are reused across
                tables. Pass a tensor rather than a Python integer so
                compiled graphs treat the count as data instead of a
                constant to specialize on.
            seqused_cols: Valid column count as a positive scalar tensor with
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                When set, only the first ``seqused_cols`` columns act as
                features; the remaining columns are padding, excluded from
                feature grouping and masked from row-wise attention. For
                :class:`~sdm.TableTensor` inputs the count refers to the
                extracted numerical block (``x.numerical.size(-1)``), not
                the table width - non-numerical columns are removed before
                masking applies. The same finite-values contract as
                ``seqused_train`` applies; out-of-range counts are clamped
                and produce degenerate predictions rather than errors.
                Unlike ``seqused_train``, the count is shared across batch
                elements.
            batch_size_limit: If set, run attention blocks in chunks of at
                most this many broadcasted batch elements to cap peak memory
                for very large batches; ``None`` disables it.

        Returns:
            The prediction for the remaining ``[..., R - R_train]`` test rows.
        """
        self._validate_seqused(seqused_train, seqused_cols)
        self._warn_seqused_cols_table(x, seqused_cols)
        if recipe is not None and (
            seqused_train is not None or seqused_cols is not None
        ):
            raise ValueError(
                "`recipe` preprocessing fits on the padded rows/targets, "
                "letting padding influence predictions in violation of the "
                "seqused contract; pass `recipe=None` together with "
                "`seqused_train`/`seqused_cols`"
            )
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        # Only forward the padding/chunking keywords when set so that
        # subclasses implementing the older hook signature keep working.
        kwargs: dict[str, Any] = {}
        if seqused_train is not None:
            kwargs["seqused_train"] = seqused_train
        if seqused_cols is not None:
            kwargs["seqused_cols"] = seqused_cols
        if batch_size_limit is not None:
            kwargs["batch_size_limit"] = batch_size_limit

        # TODO Create an ensemble dimension to process across ensemble
        # members for better efficiency.
        outs: list[Tensor] = []
        for _ in range(num_estimators):
            # TODO Iterate over Recipes instead of using a single recipe once
            # Recipe adds support for multiple recipes.
            x_i, y_i = self._preprocess(x, y, related_tables, recipe=recipe)
            out = self._forward(x_i, y_i, related_tables, cache=None, **kwargs)
            out = self._postprocess(out, recipe)
            outs.append(out)

        out = torch.stack(outs).mean(dim=0)
        table = TableTensor.from_tensor(out.clone())
        if recipe is None:
            return table.numerical
        return recipe.output.transform(table).numerical

    @_maybe_inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R_train, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        seqused_train: Tensor | None = None,  # [...]
        seqused_cols: Tensor | None = None,  # []
        batch_size_limit: int | None = None,
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
            seqused_train: Valid in-context example counts with shape ``[...]``
                and :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                When set, rows beyond the per-element count are padding and
                are masked from the cached key/value projections; subsequent
                :meth:`predict` calls reuse the count automatically. See
                :meth:`forward` for the padding contract.
            seqused_cols: Valid column count as a scalar tensor with
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                When set, columns beyond the count are padding; subsequent
                :meth:`predict` calls reuse the count and must pass ``x``
                padded to the same number of columns. See :meth:`forward`
                for the padding contract.
            batch_size_limit: If set, run attention blocks in chunks of at
                most this many broadcasted batch elements to cap peak memory
                for large batches.
        """
        self._validate_seqused(seqused_train, seqused_cols)
        self._warn_seqused_cols_table(x, seqused_cols)
        if recipe is not None and (
            seqused_train is not None or seqused_cols is not None
        ):
            raise ValueError(
                "`recipe` preprocessing fits on the padded rows/targets, "
                "letting padding influence predictions in violation of the "
                "seqused contract; pass `recipe=None` together with "
                "`seqused_train`/`seqused_cols`"
            )
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        # Only forward the padding/chunking keywords when set so that
        # subclasses implementing the older hook signature keep working.
        kwargs: dict[str, Any] = {}
        if seqused_train is not None:
            kwargs["seqused_train"] = seqused_train
        if seqused_cols is not None:
            kwargs["seqused_cols"] = seqused_cols
        if batch_size_limit is not None:
            kwargs["batch_size_limit"] = batch_size_limit

        self.clear()
        caches: list[Cache] = []
        for _ in range(num_estimators):
            # TODO Iterate over Recipes instead of using a single recipe once
            # Recipe adds support for multiple recipes.
            x_i, y_i = self._preprocess(x, y, related_tables, recipe=recipe)
            x_i = x_i[..., : y_i.size(-1), :]

            # TODO Don't store y.dtype for every estimator.
            cache = Cache({"y.dtype": y.dtype})
            if seqused_train is not None:
                cache["seqused_train"] = seqused_train
            if seqused_cols is not None:
                cache["seqused_cols"] = seqused_cols
            self._forward(x_i, y_i, related_tables, cache, **kwargs)
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

    @_maybe_inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,  # [..., R_test, C]
        related_tables: RelatedTables | None = None,
        *,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, *]
        r"""Predict unseen test examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor with shape ``[..., R_test, C]`` with
                ``R_test`` rows and ``C`` columns.
            related_tables: Additional related context provided to the model.
                If :meth:`fit` was called with ``seqused_train`` or
                ``seqused_cols``, the stored counts are reused so padded
                in-context rows stay masked; with ``seqused_cols``, ``x``
                must be padded to the same number of columns as the
                fitted rows.
            batch_size_limit: If set, run attention blocks in chunks of at
                most this many broadcasted batch elements to cap peak memory
                for large batches.

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
                dtype=cast(torch.dtype, self._caches[0]["y.dtype"]),
                device=x.device,
            )
            seqused_train = cast(Tensor | None, cache.get("seqused_train"))
            seqused_cols = cast(Tensor | None, cache.get("seqused_cols"))
            kwargs: dict[str, Any] = {}
            if seqused_train is not None:
                kwargs["seqused_train"] = seqused_train
            if seqused_cols is not None:
                kwargs["seqused_cols"] = seqused_cols
            if batch_size_limit is not None:
                kwargs["batch_size_limit"] = batch_size_limit
            # TODO Iterate over Recipes instead of using a single recipe once
            # Recipe adds support for multiple recipes.
            x_i, y_i = self._preprocess(
                x,
                y_i,
                related_tables,
                recipe=recipe,
                fit_recipe=False,
            )
            out = self._forward(x_i, y_i, related_tables, cache, **kwargs)
            out = self._postprocess(out, recipe)
            outs.append(out)

        out = torch.stack(outs).mean(dim=0)
        table = TableTensor.from_tensor(out.clone())
        if recipe is None:
            return table.numerical
        return recipe.output.transform(table).numerical

    # Helpers #################################################################

    def _warn_seqused_cols_table(
        self,
        x: Tensor | TableTensor,
        seqused_cols: Tensor | None,
    ) -> None:
        if (
            seqused_cols is not None
            and isinstance(x, TableTensor)
            and x.size(-1) != x.numerical.size(-1)
        ):
            warnings.warn(
                f"`seqused_cols` counts columns of the extracted numerical "
                f"block ({x.numerical.size(-1)} columns), but 'x' has "
                f"{x.size(-1)} table columns; non-numerical columns "
                f"(including id data) are removed before masking applies.",
                stacklevel=3,
            )

    def _validate_seqused(
        self,
        seqused_train: Tensor | None,
        seqused_cols: Tensor | None = None,
    ) -> None:
        if seqused_train is not None and seqused_train.dtype != torch.int32:
            raise ValueError(
                f"`seqused_train` must have dtype torch.int32 "
                f"(got {seqused_train.dtype})"
            )
        if seqused_cols is not None:
            if seqused_cols.dtype != torch.int32:
                raise ValueError(
                    f"`seqused_cols` must have dtype torch.int32 "
                    f"(got {seqused_cols.dtype})"
                )
            if seqused_cols.dim() != 0:
                raise ValueError(
                    f"`seqused_cols` must be a scalar tensor "
                    f"(got shape {tuple(seqused_cols.size())})"
                )

    def _preprocess(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None,
        *,
        recipe: Recipe | None = None,
        fit_recipe: bool = True,
    ) -> tuple[Tensor, Tensor]:
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

        return x, y

    def _postprocess(
        self,
        out: Tensor,  # [..., R_test, *]
        recipe: Recipe | None,
    ) -> Tensor:  # [..., R_test, *]
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

    # Abstract Methods ########################################################

    @abstractmethod
    def _forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        related_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> Tensor:  # [..., R - R_train, *]
        # Subclasses may additionally accept keyword-only `seqused_train`,
        # `seqused_cols`, and `batch_size_limit`; the public entry points
        # only forward those keywords when the caller sets them.
        pass

    @classmethod
    @abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""
