from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

import sdm.processing as sp
from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.models.kumo.tabular.table_encoder import TableEncoder
from sdm.models.tabfm.cell_embedding import CellEmbedding


class KumoTabular(ICLModel):
    """Kumo Tabular in-context classification and regression model.

    Regression standardizes targets using context rows and returns the mean
    predicted quantile on the original target scale. Classification supports
    at most ten declared target classes, including classes absent from the
    context rows.

    Args:
        device: Device of the model parameters.
        task: Prediction task.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        task: Literal["classification", "regression"] = "classification",
    ) -> None:
        super().__init__()
        self.task = task
        if task == "classification":
            num_classes, num_quantiles = 10, 0
        else:
            assert task == "regression"
            num_classes, num_quantiles = 0, 999
        self.model = _KumoTabular(
            num_classes=num_classes,
            num_quantiles=num_quantiles,
            device=device,
        )
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return Recipe(
            target=sp.StypeDispatch(
                numerical=sp.Standardize(constant_threshold=1e-8),
            ),
            output=[
                sp.ReduceEstimators(method="mean"),
                sp.TaskDispatch(
                    classification=sp.Softmax(),
                ),
            ],
        )

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
    ) -> TableTensor:  # [..., R_query, num_classes or 1]
        if x_query is None:
            assert x_context is not None
            x = x_context.numerical
        elif x_context is None:
            x = x_query.numerical
        else:
            x = torch.cat((x_context.numerical, x_query.numerical), dim=-2)

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            if self.task != "classification":
                raise ValueError(
                    f"{self.__class__.__name__!r} is initialized for task "
                    f"{self.task!r}, but received a categorical target"
                )
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            if self.task != "regression":
                raise ValueError(
                    f"{self.__class__.__name__!r} is initialized for task "
                    f"{self.task!r}, but received a numerical target"
                )
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            if self.task == "classification":
                classes = cast(Tensor, cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64
                if self.task == "classification"
                else x.dtype,
            )

        if classes is not None and len(classes) > 10:
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
            x=x,
            y=y,
            categorical_mask=categorical_mask,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
        if classes is None:
            return TableTensor(
                columns={Stype.numerical: ("pred",)},
                numerical=out.mean(dim=-1, keepdim=True),
            )
        return TableTensor(
            columns={
                Stype.numerical: [str(value) for value in classes.tolist()]
            },
            numerical=out[..., : len(classes)],
        )


class _KumoTabular(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_classes = num_classes

        self.cell_embedding = CellEmbedding(
            channels=128,
            group_size=3,
            num_frequencies=32,
            **factory_kwargs,
        )
        self.y_encoder = Linear(
            num_classes or 1,
            128,
            bias=num_classes > 0,
            **factory_kwargs,
        )
        self.table_encoder = TableEncoder(**factory_kwargs)
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=512,
            num_layers=12,
            num_heads=8,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        R_train = y.size(-1)
        x = self.cell_embedding(x, categorical_mask)

        if y.numel() > 0:
            if self.num_classes > 0:
                y_emb = F.one_hot(y.long(), num_classes=self.num_classes)
            else:
                y_emb = y.unsqueeze(-1)
            y_emb = self.y_encoder(y_emb.to(self.y_encoder.weight.dtype)).to(
                x.dtype
            )
            x = torch.cat(
                (
                    x[..., :R_train, :, :] + y_emb.unsqueeze(-2),
                    x[..., R_train:, :, :],
                ),
                dim=-3,
            )

        x = self.table_encoder(x, R_train, cache=cache)
        return self.icl_block(
            x=x,
            y=y,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
