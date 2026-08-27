from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models.base import ICLModel
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.kumo.tabular.recipe import default_recipe


class KumoTabular(ICLModel):
    """Kumo Tabular in-context classification model.

    The default recipe aligns and converts categorical columns but expects
    otherwise preprocessed, finite features. The model supports at most ten
    declared target classes, including classes absent from the context rows.

    Args:
        device: Device of the model parameters.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.categorical}
    )
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.model = _KumoTabular(
            num_classes=10,
            num_quantiles=0,
            device=device,
        )
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def forward(self, *args: Any, **kwargs: Any) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        x_context = kwargs["x_context"] if "x_context" in kwargs else args[0]
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        kwargs["_categorical_columns"] = frozenset(
            x_context.columns[Stype.categorical]
        )
        return super().forward(*args, **kwargs)

    def fit(self, *args: Any, **kwargs: Any) -> None:
        r""":meta private:"""  # noqa: D415
        x = kwargs["x"] if "x" in kwargs else args[0]
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        kwargs["_categorical_columns"] = frozenset(
            x.columns[Stype.categorical]
        )
        return super().fit(*args, **kwargs)

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        batch_size_limit: int | None = None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes]
        if x_query is None:
            assert x_context is not None
            x = x_context.numerical
        elif x_context is None:
            x = x_query.numerical
        else:
            x = torch.cat((x_context.numerical, x_query.numerical), dim=-2)

        if y_context is not None:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        else:
            assert cache is not None
            classes = cast(Tensor, cache["classes"])
            y = x.new_empty((*x.size()[:-2], 0), dtype=torch.int64)

        if len(classes) > 10:
            raise ValueError(
                f"{self.__class__.__name__!r} only supports up to 10 classes "
                f"(got {len(classes)})"
            )

        if cache is None or cache.is_recording:
            assert x_context is not None
            categorical_columns = cast(
                frozenset[str], kwargs["_categorical_columns"]
            )
            categorical_mask = torch.tensor(
                [
                    column in categorical_columns
                    for column in x_context.columns[Stype.numerical]
                ],
                device=x.device,
                dtype=torch.bool,
            )
            if cache is not None:
                cache["categorical_mask"] = categorical_mask
        else:
            categorical_mask = cast(Tensor, cache["categorical_mask"])
        categorical_mask = categorical_mask.expand(*x.size()[:-2], -1)

        out = self.model(
            x,
            y,
            categorical_mask,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
        return TableTensor(
            columns={
                Stype.numerical: [str(value) for value in classes.tolist()]
            },
            numerical=out[..., : len(classes)],
        )
