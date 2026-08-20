import contextlib
import functools
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import torch

from sdm import RelatedTables, TableTensor

_F = TypeVar("_F", bound=Callable[..., Any])


class Callback:
    """Base class for callbacks applied to one model call.

    Lifecycle hooks run automatically around
    :meth:`~sdm.models.ICLModel.forward` and
    :meth:`~sdm.models.ICLModel.predict`. The start hook runs once per
    attempted call, the end hook runs once after a successful call, and
    preprocessing hooks run once per ensemble member.

    Callbacks supplied together run in sequence order, each preprocessing
    result is passed to the next callback, and their execution contexts enter
    before lifecycle hooks in sequence order and exit in reverse order.
    """

    def execution_context(
        self,
        model: torch.nn.Module,
    ) -> contextlib.AbstractContextManager[None]:
        """Return a fresh context manager for one model call.

        The context encloses all callback hooks and model execution and must
        not suppress exceptions.

        Args:
            model: Model receiving the callback.

        Returns:
            Context manager for the public model call.
        """
        return contextlib.nullcontext()

    def on_forward_start(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run before input validation and preprocessing.

        Args:
            model: Model receiving the callback.
            args: Model inputs in public signature order, including defaulted
                values.
            kwargs: Model options by name, including defaulted values.
        """

    def on_forward_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
    ) -> None:
        """Run after successful output postprocessing.

        Args:
            model: Model receiving the callback.
            prediction: Fully processed prediction returned by the public
                model call.
        """

    def on_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        """Transform one ensemble member after preprocessing.

        Args:
            model: Model receiving the callback.
            x: Preprocessed query table for the ensemble member.
            related_tables: Preprocessed related query tables for the ensemble
                member, if any.

        Returns:
            Query table and related tables passed to the next callback or the
            model.
        """
        return x, related_tables


def _callback_contexts(function: _F) -> _F:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        callbacks = cast(
            Sequence[Callback] | None,
            kwargs.get("callbacks"),
        )
        if not callbacks:
            return function(*args, **kwargs)

        model = cast(torch.nn.Module, args[0])
        with contextlib.ExitStack() as stack:
            for callback in callbacks:
                stack.enter_context(callback.execution_context(model))
            return function(*args, **kwargs)

    return cast(_F, wrapper)
