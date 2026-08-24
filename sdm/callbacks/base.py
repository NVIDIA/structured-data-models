from typing import Any

import torch

from sdm import RelatedTables, TableTensor


class Callback:
    """Base class for callbacks applied to one model call.

    Lifecycle hooks run automatically around
    :meth:`~sdm.models.ICLModel.forward` and
    :meth:`~sdm.models.ICLModel.predict`. The start hook runs once per
    attempted call, the end hook runs once after a successful call, and
    preprocessing hooks run once per ensemble member.

    Callbacks supplied together run in sequence order, and each preprocessing
    result is passed to the next callback.
    """

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
