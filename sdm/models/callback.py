import torch

from sdm import RelatedTables, TableTensor


class Callback:
    """Base class for callbacks attached to one model call.

    Lifecycle hooks run automatically around
    :meth:`~sdm.models.ICLModel.forward` and
    :meth:`~sdm.models.ICLModel.predict`.

    Callbacks supplied together run in sequence order.
    """

    def on_forward_start(self, model: torch.nn.Module, /) -> None:
        """Run before a model forward pass.

        Args:
            model: Model receiving the callback.
        """

    def on_forward_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
        /,
    ) -> None:
        """Run after a successful model forward pass.

        Args:
            model: Model receiving the callback.
            prediction: Fully processed value returned by the public
                :meth:`~sdm.models.ICLModel.forward` call.
        """

    def on_predict_start(self, model: torch.nn.Module, /) -> None:
        """Run before a fitted model prediction.

        Args:
            model: Model receiving the callback.
        """

    def on_predict_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
        /,
    ) -> None:
        """Run after a successful fitted model prediction.

        Args:
            model: Model receiving the callback.
            prediction: Fully processed value returned by the public
                :meth:`~sdm.models.ICLModel.predict` call.
        """

    def on_after_preprocessing(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
        /,
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
