from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ModuleDict

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.models.kumo.tabular.row_embedding import RowEmbedding
from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.tensor.table import TableSchema

MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "small": {
        "cell_channels": 128,
        "num_embedding_layers": 4,
        "num_embedding_heads": 4,
        "num_inducing_points": 128,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 12,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": None,
    },
    "large": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 24,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": 2,
    },
}


class KumoTabular(ICLModel):  # noqa: D101
    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None = None,
        size: Literal["small", "large"] = "large",
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=task)

        self.models: ModuleDict[TaskLike, torch.nn.Module] = ModuleDict()
        for task in self.tasks:
            self.models[task] = _KumoTabular(
                num_classes=10 if task == Task.classification else 0,
                num_quantiles=999 if task == Task.regression else 0,
                device="meta" if pretrained else device,
                **MODEL_KWARGS[size],
            )

        if pretrained:
            self.models[task] = self._load_from_pretrained(size, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        size: Literal["small", "large"],
        device: torch.device | str | None,
    ) -> _KumoTabular:
        device = torch.get_default_device() if device is None else device

        for task, model in self.models.items():
            if task == Task.classification:
                filename = f"{size}/classifier.pt"
            else:
                assert task == Task.regression
                filename = f"{size}/regressor.pt"

            path = download_checkpoint(
                repo_id="nvidia/Kumo-Tabular",
                filename=filename,
                revision="v1.0.1",
            )
            ckpt = torch.load(path, map_location=device, weights_only=True)
            ckpt = remap_ckpt(ckpt, is_classifier=task == Task.classification)
            model.load_state_dict(ckpt, assign=True)

        return model

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
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes or 999]

        if x_query is None and x_context is not None:
            x = x_context.numerical
        elif x_context is None and x_query is not None:
            x = x_query.numerical
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            classes = cast(Tensor | None, cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        if classes is not None and len(classes) > 10:
            raise ValueError(
                f"{self.__class__.__name__!r} only supports up to 10 classes "
                f"(got {len(classes)})"
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

        task = Task.classification if classes is not None else Task.regression
        out = self.models[task](x, y, categorical_mask, cache=cache)

        if classes is None:
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out.sort(dim=-1)[0],
            )
        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out[..., : len(classes)],
        )


class _KumoTabular(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        cell_channels: int = 128,
        num_embedding_layers: int = 4,
        num_embedding_heads: int = 4,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 4,
        icl_channels: int = 512,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        num_icl_key_value_heads_for_query: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.cell_embedding = CellEmbedding(
            channels=cell_channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )
        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=cell_channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        if cell_channels * num_readout_tokens != icl_channels:
            self.row_project = Linear(
                cell_channels * num_readout_tokens,
                icl_channels,
                **factory_kwargs,
            )
        else:
            self.row_project = Identity()
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=icl_channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            num_key_value_heads_for_query=num_icl_key_value_heads_for_query,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        x = self.cell_embedding(x, categorical_mask, batch_size_limit="auto")
        return x.new_empty(x.size(-3) - y.size(-1), 999)
        return x
        x = self.row_embedding(x, y, cache=cache)
        x = self.row_project(x)
        return self.icl_block(x=x, y=y, cache=cache)


def remap_ckpt(  # noqa: D103
    ckpt: dict[str, Tensor],
    is_classifier: bool,
) -> dict[str, Tensor]:

    out: dict[str, Tensor] = {}
    for key, value in ckpt.items():
        if key == "y_encoder.weight":
            if is_classifier:
                out["row_embedding.y_emb.weight"] = (
                    value.T + ckpt["y_encoder.bias"]
                )
            else:
                out["row_embedding.y_lin.weight"] = value
            continue

        if is_classifier and key == "y_encoder.bias":
            continue

        if key == "table_encoder.cls_tokens":
            out["row_embedding.readout_token"] = value
            continue

        if key.startswith("table_encoder."):
            key = key.removeprefix("table_encoder.")
            out[f"row_embedding.{key}"] = value
            continue

        if key.startswith("cls_downproject."):
            key = key.removeprefix("cls_downproject.")
            out[f"row_project.{key}"] = value
            continue

        out[key] = value

    return out
