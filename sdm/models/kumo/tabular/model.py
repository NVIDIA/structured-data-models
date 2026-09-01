from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleDict

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from sdm.cache import Cache
from sdm.models import ICLModel, TabICLv2
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.models.kumo.tabular.table_encoder import TableEncoder
from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.tensor.table import TableSchema

_CHECKPOINT_FILE = {
    Task.classification: "classifier.pt",
    Task.regression: "regressor.pt",
}


def _architecture_kwargs(state: Mapping[str, Tensor]) -> dict[str, Any]:
    channels = state["cell_embedding.num_lin.weight"].size(0)
    if channels == 128:
        return {
            "channels": 128,
            "num_embedding_layers": 4,
            "num_embedding_col_heads": 4,
            "num_embedding_row_heads": 4,
            "num_inducing_points": 128,
            "group_size": 3,
            "num_frequencies": 32,
            "num_readout_tokens": 4,
            "downproject_cls_factor": 1.0,
            "num_icl_layers": 12,
            "num_icl_heads": 8,
            "icl_num_kv_heads_test": None,
        }
    if channels == 256:
        return {
            "channels": 256,
            "num_embedding_layers": 6,
            "num_embedding_col_heads": 4,
            "num_embedding_row_heads": 4,
            "num_inducing_points": 256,
            "group_size": 3,
            "num_frequencies": 32,
            "num_readout_tokens": 4,
            "downproject_cls_factor": 0.5,
            "num_icl_layers": 24,
            "num_icl_heads": 8,
            "icl_num_kv_heads_test": 2,
        }
    raise ValueError(f"Unsupported KumoTabular checkpoint width: {channels}")


# TODO: Add model documentation.
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
            if pretrained:
                self.models[task] = self._load_from_pretrained(
                    task,
                    size,
                    device,
                )
            else:
                self.models[task] = _KumoTabular(
                    num_classes=10 if task == Task.classification else 0,
                    num_quantiles=999 if task == Task.regression else 0,
                    device=device,
                )

        self.eval()

    def _load_from_pretrained(
        self,
        task: Task,
        size: Literal["small", "large"],
        device: torch.device | str | None,
    ) -> _KumoTabular:
        device = torch.get_default_device() if device is None else device
        checkpoint_path = download_checkpoint(
            repo_id="nvidia/Kumo-Tabular",
            filename=f"{size}/{_CHECKPOINT_FILE[task]}",
            revision="v1.0.1",
        )
        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model = _KumoTabular(
            num_classes=10 if task == Task.classification else 0,
            num_quantiles=999 if task == Task.regression else 0,
            **_architecture_kwargs(state),
            device="meta",
        )
        model.load_state_dict(state, strict=True, assign=True)
        return model

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        # TODO: Define the default recipe.
        return TabICLv2.default_recipe()

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
        batch_size_limit: int | None = None,
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
        out = self.models[task](
            x=x,
            y=y,
            categorical_mask=categorical_mask,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
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
        channels: int = 128,
        num_embedding_layers: int = 4,
        num_embedding_col_heads: int = 4,
        num_embedding_row_heads: int = 4,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 4,
        downproject_cls_factor: float = 1.0,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        icl_num_kv_heads_test: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_classes = num_classes
        cls_channels = num_readout_tokens * channels
        icl_channels = round(cls_channels * downproject_cls_factor)
        if icl_channels < 1:
            raise ValueError(
                "`downproject_cls_factor` must yield at least one ICL channel"
            )

        self.cell_embedding = CellEmbedding(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )
        self.y_encoder = Linear(
            in_features=num_classes or 1,
            out_features=channels,
            bias=num_classes > 0,
            **factory_kwargs,
        )
        self.table_encoder = TableEncoder(
            channels=channels,
            num_layers=num_embedding_layers,
            num_col_heads=num_embedding_col_heads,
            num_row_heads=num_embedding_row_heads,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )

        self.cls_downproject: Linear | None = None
        if icl_channels != cls_channels:
            self.cls_downproject = Linear(
                cls_channels,
                icl_channels,
                **factory_kwargs,
            )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=icl_channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            num_key_value_heads_test=icl_num_kv_heads_test,
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
        if self.cls_downproject is not None:
            x = self.cls_downproject(x)
        return self.icl_block(
            x=x,
            y=y,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
