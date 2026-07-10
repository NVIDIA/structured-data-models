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


def _validate_batch_size_limit(batch_size_limit: int | None) -> None:
    if batch_size_limit is not None and batch_size_limit <= 0:
        raise ValueError("`batch_size_limit` must be positive")


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
            batch_size_limit: Maximum number of broadcast attention batch
                elements processed at once. ``None`` disables batch chunking.

        Returns:
            The prediction for the remaining ``[..., R - R_train]`` test rows.
        """
        _validate_batch_size_limit(batch_size_limit)
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        x, y = self._preprocess(x, y, recipe=recipe)
        # TODO Create an ensemble dimension to process across ensemble
        # members for better efficiency.
        outs: list[Tensor] = []
        for _ in range(num_estimators):
            outs.append(
                self._forward(
                    x,
                    y,
                    related_tables,
                    cache=None,
                    batch_size_limit=batch_size_limit,
                )
            )
        return self._postprocess(outs, recipe)

    @torch.inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R_train, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
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
            batch_size_limit: Maximum number of broadcast attention batch
                elements processed at once. ``None`` disables batch chunking.
        """
        _validate_batch_size_limit(batch_size_limit)
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        self.clear()
        x, y = self._preprocess(x, y, recipe=recipe)
        x = x[..., : y.size(-1), :]
        caches: list[Cache] = []
        for _ in range(num_estimators):
            # TODO Don't store y.dtype for every estimator.
            cache = Cache({"y.dtype": y.dtype})
            self._forward(
                x,
                y,
                related_tables,
                cache,
                batch_size_limit=batch_size_limit,
            )
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
        *,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, *]
        r"""Predict unseen test examples.

        .. note::

            This method requires a prior call to :meth:`fit`.
            A recipe passed to :meth:`fit` is reused to transform ``x`` and
            to postprocess predictions.

        Args:
            x: The feature tensor with shape ``[..., R_test, C]`` with
                ``R_test`` rows and ``C`` columns.
            related_tables: Additional related context provided to the model.
            batch_size_limit: Maximum number of broadcast attention batch
                elements processed at once. ``None`` disables batch chunking.

        Returns:
            The prediction for ``[..., R_test]`` test rows.
        """
        _validate_batch_size_limit(batch_size_limit)
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
        y = torch.empty(
            (*x.size()[:-2], 0),
            dtype=cast(torch.dtype, self._caches[0]["y.dtype"]),
            device=x.device,
        )
        x, y = self._preprocess(x, y, recipe=recipe, fit_recipe=False)
        outs: list[Tensor] = []
        for cache in self._caches:
            outs.append(
                self._forward(
                    x,
                    y,
                    related_tables,
                    cache,
                    batch_size_limit=batch_size_limit,
                )
            )
        return self._postprocess(outs, recipe)

    # Helpers #################################################################

    # FIXME: Fix TableTensor to support inference mode.
    @torch.inference_mode(False)
    def _preprocess(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        recipe: Recipe | None = None,
        fit_recipe: bool = True,
    ) -> tuple[Tensor, Tensor]:
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
        outs: list[Tensor],  # E x [..., R_test, *]
        recipe: Recipe | None,
    ) -> Tensor:  # [..., R_test, *]
        # Average predictions across ensemble members:
        out = torch.stack(outs).mean(dim=0)
        if recipe is None:
            return out

        if not isinstance(recipe.target, InvertibleMixin):
            raise ValueError(
                f"Expected the target steps of 'recipe' to support "
                f"'inverse_transform' to map predictions back to the "
                f"original target space "
                f"(got '{recipe.target.__class__.__name__}')"
            )

        # Recipe steps run in normal mode since 'TableTensor' does not
        # support structural ops on inference tensors. Cloning the prediction
        # moves it out of inference mode:
        with torch.inference_mode(False):
            table = TableTensor.from_tensor(out.clone())
            table = recipe.target.inverse_transform(table)
            table = recipe.output.transform(table)
            return table.numerical

    # Abstract Methods ########################################################

    @abstractmethod
    def _forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        related_tables: RelatedTables | None,
        cache: Cache | None,
        *,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R - R_train, *]
        pass

    @classmethod
    @abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""
