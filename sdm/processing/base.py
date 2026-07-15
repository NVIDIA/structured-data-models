import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import torch
from typing_extensions import Self

from sdm.processing.context import RecipeContext
from sdm.stype import Stype
from sdm.tensor import TableTensor

SupportedStypes: TypeAlias = frozenset[Stype]


class Processor(torch.nn.Module, abc.ABC):
    """Fittable, table-in/table-out transform.

    Subclass and implement ``_transform`` (the transform operation). Override
    ``_fit`` to learn state from a :class:`TableTensor` (the default is a
    no-op). For an inverse, also mix in :class:`InvertibleMixin` and implement
    ``_inverse_transform``. Set ``requires_fit = False`` for stateless
    processors that can safely run without a prior ``fit`` call. Set
    ``supported_stypes`` for processors that support non-numerical columns.
    Context-aware output processors additionally set ``required_context`` and
    implement ``_transform_with_context``.
    """

    supported_stypes: ClassVar[SupportedStypes]
    required_context: ClassVar[frozenset[str]] = frozenset()
    requires_fit: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def _check_supported_stypes(self, table: TableTensor) -> None:
        supported_stypes = self.supported_stypes
        for stype, columns in table.columns.items():
            if stype not in supported_stypes and len(columns) > 0:
                # TODO: Include all invalid columns in the error message
                raise ValueError(
                    f"'{self.__class__.__name__}' does not support "
                    f"'{stype.value}' columns."
                )

    def _check_is_fitted(self) -> None:
        if self.requires_fit and not self._fitted:
            raise RuntimeError(
                f"'{self.__class__.__name__}' is not fitted; "
                "call 'fit()' before."
            )

    def _fit(self, table: TableTensor) -> None:
        pass

    @abc.abstractmethod
    def _transform(self, table: TableTensor) -> TableTensor:
        pass

    def _transform_with_context(
        self,
        table: TableTensor,
        context: Sequence[RecipeContext] | None,
    ) -> TableTensor:
        return self._transform(table)

    def _check_context(
        self,
        context: Sequence[RecipeContext] | None,
    ) -> None:
        if len(self.required_context) == 0:
            return
        if context is None or len(context) == 0:
            required = ", ".join(sorted(self.required_context))
            raise RuntimeError(
                f"'{self.__class__.__name__}' requires RecipeContext values "
                f"{required}."
            )

        for estimator_context in context:
            missing = tuple(
                name
                for name in sorted(self.required_context)
                if getattr(estimator_context, name, None) is None
            )
            if len(missing) > 0:
                values = ", ".join(missing)
                raise RuntimeError(
                    f"'{self.__class__.__name__}' requires populated "
                    f"RecipeContext values {values} for estimator "
                    f"{estimator_context.estimator_index}."
                )

    def forward(
        self,
        table: TableTensor,
        *,
        context: Sequence[RecipeContext] | None = None,
    ) -> TableTensor:
        """Alias of :meth:`~Processor.transform`.

        This is the :class:`torch.nn.Module` entry point, so
        ``processor(table)`` and ``processor.transform(table)`` share the same
        fitted-state checks.

        Args:
            table: Table to transform.
            context: Optional fitted contexts in estimator order.

        Returns:
            Transformed table.
        """
        return self.transform(table, context=context)

    def fit(self, table: TableTensor) -> Self:
        """Fit the processor on ``table`` and return it.

        Args:
            table: Feature table used to compute the processor state.

        Returns:
            This processor.
        """
        self._check_supported_stypes(table)
        if self.requires_fit:
            self._fit(table)
            self._fitted = True
        return self

    def transform(
        self,
        table: TableTensor,
        *,
        context: Sequence[RecipeContext] | None = None,
    ) -> TableTensor:
        """Transform ``table`` using the fitted processor.

        Args:
            table: Table to transform.
            context: Optional fitted contexts in estimator order.

        Returns:
            Transformed table.
        """
        self._check_supported_stypes(table)
        self._check_is_fitted()
        self._check_context(context)
        return self._transform_with_context(table, context)

    def fit_transform(self, table: TableTensor) -> TableTensor:
        """Fit on ``table`` and return the transformed result.

        Args:
            table: Feature table to fit on and transform.

        Returns:
            Transformed table.
        """
        return self.fit(table).transform(table)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}()"


class InvertibleMixin(abc.ABC):
    """Adds ``inverse_transform`` to a :class:`Processor`.

    Combine with :class:`Processor` and implement ``_inverse_transform``,
    e.g. ``class StandardScale(Processor, InvertibleMixin): ...``.
    """

    @abc.abstractmethod
    def _inverse_transform(self, table: TableTensor) -> TableTensor: ...

    def inverse_transform(self, table: TableTensor) -> TableTensor:
        """Invert the transform of ``table`` using the fitted processor.

        Args:
            table: Table in transformed space.

        Returns:
            Table mapped back to the original processor space.
        """
        self._check_is_fitted()
        return self._inverse_transform(table)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...
