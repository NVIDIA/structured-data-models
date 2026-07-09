import contextlib
import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.processing import Recipe


class BaseModel(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`BaseModel` defines the public inferface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    supports_related_tables: ClassVar[bool]

    def __init__(self):
        super().__init__()

        # One cache per ensemble member.
        self._caches: list[Cache] | None = None

    def forward(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None = None,
        *,
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
            num_estimators: The number of estimators for ensembling.

        Returns:
            The prediction for the remaining ``[..., R - R_train]`` test rows.
        """
        # `torch.inference_mode` cannot be traced by `torch.compile` (a
        # decorated forward fails `fullgraph=True` compilation at the
        # AOTAutograd stage), so compiled callers enter it around the
        # compiled call instead.
        context = (
            contextlib.nullcontext()
            if torch.compiler.is_compiling()
            else torch.inference_mode()
        )
        with context:
            if not self.supports_related_tables and related_tables is not None:
                warnings.warn(
                    f"'{self.__class__.__name__}' does not support "
                    f"related tables",
                    stacklevel=2,
                )
                related_tables = None

            x, y = self._preprocess(x, y)
            # TODO Create an ensemble dimension to process across ensemble
            # members for better efficiency.
            outs: list[Tensor] = []
            for _ in range(num_estimators):
                outs.append(self._forward(x, y, related_tables, cache=None))
            return torch.stack(outs).mean(dim=0)

    @torch.inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R_train, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        related_tables: RelatedTables | None = None,
        *,
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
            num_estimators: The number of estimators for ensembling.
        """
        if not self.supports_related_tables and related_tables is not None:
            warnings.warn(
                f"'{self.__class__.__name__}' does not support related tables",
                stacklevel=2,
            )
            related_tables = None

        self.clear()
        x, y = self._preprocess(x, y)
        x = x[..., : y.size(-1), :]
        caches: list[Cache] = []
        for _ in range(num_estimators):
            # TODO Don't store y.dtype for every estimator.
            cache = Cache({"y.dtype": y.dtype})
            self._forward(x, y, related_tables, cache)
            cache.freeze()
            caches.append(cache)
        self._caches = caches

    def clear(self) -> None:
        r"""Clears cached in-context examples."""
        self._caches = None

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

        y = torch.empty(
            (*x.size()[:-2], 0),
            dtype=cast(torch.dtype, self._caches[0]["y.dtype"]),
            device=x.device,
        )
        x, y = self._preprocess(x, y)
        outs: list[Tensor] = []
        for cache in self._caches:
            outs.append(self._forward(x, y, related_tables, cache))
        return torch.stack(outs).mean(dim=0)

    # Helpers #################################################################

    def _preprocess(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
    ) -> tuple[Tensor, Tensor]:
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
