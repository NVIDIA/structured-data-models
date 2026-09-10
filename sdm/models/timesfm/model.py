from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import torch

from sdm import Recipe, RelatedTables, Stype, TableTensor, TaskLike
from sdm.cache import Cache
from sdm.models.base import ICLModel
from sdm.models.timesfm.recipe import default_recipe


class TimesFM(ICLModel):
    r"""TimesFM adapter for causal time-series forecasting.

    This adapter interprets the :class:`~sdm.models.ICLModel` boundary as a
    forecasting boundary: ``x_context`` and ``y_context`` describe the observed
    history, while ``x_query`` describes the future rows to forecast.

    Args:
        past_only_columns: Feature columns available only for the historical
            context.
        past_future_columns: Feature columns available for both context and
            future query rows. If ``None``, all numerical feature columns not
            listed in ``past_only_columns`` are treated as past-future
            covariates after recipe preprocessing.
        return_quantiles: Whether model output should include all TimesFM
            quantiles in addition to the median forecast.
        median_quantile_index: Quantile index used as the point forecast.
        task: The task to initialize. Only regression is supported.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.datetime}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        *,
        past_only_columns: Sequence[str] = (),
        past_future_columns: Sequence[str] | None = None,
        return_quantiles: bool = False,
        median_quantile_index: int = 4,
        task: TaskLike | Iterable[TaskLike] | None = "regression",
    ) -> None:
        super().__init__(task=task)

        self.past_only_columns = tuple(past_only_columns)
        self.past_future_columns = (
            None if past_future_columns is None else tuple(past_future_columns)
        )
        self.return_quantiles = return_quantiles
        self.median_quantile_index = median_quantile_index

        if self.past_future_columns is not None:
            overlap = set(self.past_only_columns) & set(
                self.past_future_columns
            )
            if overlap:
                columns = ", ".join(sorted(overlap))
                raise ValueError(
                    "'past_only_columns' and 'past_future_columns' must not "
                    f"overlap; found {columns}"
                )

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, 1] or [..., R_query, Q + 1]
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def _covariate_columns(
        self,
        x: TableTensor,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        numerical_columns = x.columns[Stype.numerical]

        past_only_columns = self.past_only_columns
        if self.past_future_columns is None:
            past_only = set(past_only_columns)
            past_future_columns = tuple(
                column
                for column in numerical_columns
                if column not in past_only
            )
        else:
            past_future_columns = self.past_future_columns

        self._validate_numerical_columns(
            x=x,
            columns=past_only_columns,
            role="past-only",
        )
        self._validate_numerical_columns(
            x=x,
            columns=past_future_columns,
            role="past-future",
        )
        return past_only_columns, past_future_columns

    def _validate_numerical_columns(
        self,
        *,
        x: TableTensor,
        columns: Sequence[str],
        role: str,
    ) -> None:
        numerical_columns = set(x.columns[Stype.numerical])
        missing = [
            column for column in columns if column not in numerical_columns
        ]
        if missing:
            names = ", ".join(repr(column) for column in missing)
            raise ValueError(
                f"{role} covariate columns must be numerical after recipe "
                f"preprocessing; missing {names}"
            )
