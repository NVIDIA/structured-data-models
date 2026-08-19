from typing import Any

import torch

from sdm import RelatedTables, TableTensor


class Callback:
    """Base class for callbacks attached to one model call.

    Lifecycle hooks run automatically around
    :meth:`~sdm.models.ICLModel.forward` and
    :meth:`~sdm.models.ICLModel.predict`.

    Callbacks supplied together run in sequence order.
    """

    def on_forward_start(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run before a model forward or prediction pass.

        Args:
            model: Model receiving the callback.
            args: Positional inputs to the public model call.
            kwargs: Keyword inputs to the public model call.
        """

    def on_forward_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
    ) -> None:
        """Run after a successful model forward or prediction pass.

        Args:
            model: Model receiving the callback.
            prediction: Fully processed value returned by the public model
                call.
        """

    def on_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        """Run after preprocessing and before each model execution.

        Args:
            model: Model receiving the callback.
            x: Preprocessed query table.
            related_tables: Preprocessed related query tables, if any.

        Returns:
            Query inputs passed to the next callback or model.
        """
        return x, related_tables
