from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models.base import ICLModel
from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.models.tabfm.icl import ICLBlock
from sdm.models.tabfm.row_embedding import RowEmbedding
from sdm.tensor.table import TableSchema


class TabFM(ICLModel):
    r"""The Google TabFM tabular foundation model.

    Args:
        task: The prediction task.
        checkpoint_path: The local checkpoint path.
        device: The device.
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
        task: Literal["classification", "regression"],
        checkpoint_path: str | Path | None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        assert checkpoint_path is None

        self.task = task

        self.model = _TabFM(
            num_classes=10 if task == "classification" else 0,
            device=device,
        )

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return Recipe(  # TODO Replace with final recipe.
            features=[
                sp.StypeDispatch(categorical=sp.ToNumerical()),
                sp.ShuffleColumns(),
            ]
        )

    def forward(self, *args: Any, **kwargs: Any) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        x_context = kwargs["x_context"] if "x_context" in kwargs else args[0]
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        kwargs["_schema"] = x_context.schema
        return super().forward(*args, **kwargs)

    def fit(self, *args: Any, **kwargs: Any) -> None:
        r""":meta private:"""  # noqa: D415
        x = kwargs["x"] if "x" in kwargs else args[0]
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        kwargs["_schema"] = x.schema
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
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes or 1]

        if x_query is None and x_context is not None:
            x = x_context.numerical
        elif x_context is None and x_query is not None:
            x = x_query.numerical
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        y: Tensor | None = None
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
        elif cache is not None:
            classes = cast(Tensor | None, cache["classes"])

        if classes is not None and len(classes) > 10:
            raise ValueError(
                f"{self.__class__.__name__!r} only supports up to 10 classes "
                f"(got {len(classes)})"
            )

        if y is None:
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        if cache is None or cache.is_recording:
            assert x_context is not None
            schema: TableSchema = kwargs["_schema"]
            categorical_columns = set(schema.columns[Stype.categorical])
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

        out = self.model(x, y, categorical_mask, cache=cache)

        if classes is None:
            return TableTensor(
                columns={Stype.numerical: ["pred"]},
                numerical=out,
            )

        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out[..., : len(classes)],
        )


class _TabFM(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int = 256,
        num_embedding_layers: int = 3,
        num_embedding_repeats: int = 2,
        num_embedding_col_heads: int = 4,
        num_embedding_row_heads: int = 8,
        num_inducing_points: int = 256,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 8,
        num_icl_layers: int = 24,
        num_icl_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.cell_embedding = CellEmbedding(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )
        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_embedding_layers,
            num_repeats=num_embedding_repeats,
            num_col_heads=num_embedding_col_heads,
            num_row_heads=num_embedding_row_heads,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=max(1, num_classes),
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, 1 or num_classes]
        x = self.cell_embedding(x, categorical_mask)
        x = self.row_embedding(x, y, cache=cache)
        return self.icl_block(x, y, cache=cache)
