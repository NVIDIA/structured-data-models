from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models.base import ICLModel
from sdm.models.tabfm.checkpoint import (
    _load_tabfm_v1_0_0,
    _load_tabfm_v1_0_0_from_huggingface,
)
from sdm.models.tabfm.recipe import _categorical_mask, default_recipe
from sdm.tensor.table import TableSchema

_Task = Literal["classification", "regression"]
_RAW_SCHEMA = "_tabfm_raw_schema"


class TabFM(ICLModel):
    r"""Google TabFM v1.0.0 for zero-shot tabular prediction.

    This initial wrapper accepts unbatched two-dimensional tables. Calling
    :meth:`fit` records transformed context tensors in an SDM cache, while
    :meth:`predict` recomputes the full neural core.
    Custom feature recipes must preserve original feature column names.
    See :doc:`/api/models` for released-weight usage terms.

    Args:
        task: Whether to load the classification or regression checkpoint.
        checkpoint_path: Local official-format SafeTensor checkpoint. If
            omitted, download the selected checkpoint from the pinned Google
            Hugging Face revision.
        cache_dir: Hugging Face cache directory. Only used for Hub loading.
        local_files_only: Restrict Hub loading to locally cached files.
        device: Device on which to load the model.
        dtype: Model compute dtype.
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
        task: _Task,
        *,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.task = task
        if checkpoint_path is None:
            self.model = _load_tabfm_v1_0_0_from_huggingface(
                task=task,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                device=device,
                dtype=dtype,
            )
        else:
            if cache_dir is not None or local_files_only:
                raise ValueError(
                    "cache_dir and local_files_only only apply to Hub loading"
                )
            self.model = _load_tabfm_v1_0_0(
                checkpoint_path,
                task=task,
                device=device,
                dtype=dtype,
            )
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def forward(
        self,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> TableTensor:
        r"""Run zero-shot prediction on context and query rows."""
        x_context = self._features(x_context)
        y_context = self._target(y_context)
        x_query = self._features(x_query)
        if x_context.schema != x_query.schema:
            raise ValueError(
                "Expected context and query features to share the same schema"
            )
        kwargs[_RAW_SCHEMA] = x_context.schema
        return super().forward(
            x_context,
            y_context,
            x_query,
            related_context_tables,
            related_query_tables,
            recipe=recipe,
            num_estimators=num_estimators,
            generator=generator,
            **kwargs,
        )

    def fit(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Record transformed context for repeated prediction."""
        x = self._features(x)
        y = self._target(y)
        kwargs[_RAW_SCHEMA] = x.schema
        super().fit(
            x,
            y,
            related_tables,
            recipe=recipe,
            num_estimators=num_estimators,
            generator=generator,
            **kwargs,
        )

    def predict(
        self,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:
        r"""Predict query rows after :meth:`fit`."""
        x = self._features(x)
        if self._cache is not None:
            kwargs = cast(dict[str, Any], self._cache["kwargs"])
            if cast(TableSchema, kwargs[_RAW_SCHEMA]) != x.schema:
                raise ValueError(
                    "Expected context and query features to share the same "
                    "schema"
                )
        return super().predict(x, related_tables)

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        del related_context_tables, related_query_tables, generator
        raw_schema = cast(TableSchema, kwargs[_RAW_SCHEMA])
        transformed = x_context if x_context is not None else x_query
        assert transformed is not None
        self._validate_transformed(transformed, raw_schema)

        categorical_columns = raw_schema.columns[Stype.categorical]
        if cache is not None and cache.is_recording:
            assert x_context is not None
            assert y_context is not None
            targets, _ = self._target_data(y_context)
            cache["tabfm_features"] = x_context.numerical
            cache["tabfm_targets"] = targets
            return x_context

        if cache is not None:
            assert cache.is_replaying
            assert x_query is not None
            context_features = cast(Tensor, cache["tabfm_features"])
            context_targets = cast(Tensor, cache["tabfm_targets"])
            classes = cast(Tensor | None, cache["classes"])
        else:
            assert x_context is not None
            assert y_context is not None
            assert x_query is not None
            context_features = x_context.numerical
            context_targets, classes = self._target_data(y_context)

        assert x_query is not None
        categorical_mask = _categorical_mask(x_query, categorical_columns)
        num_context = context_features.size(0)
        num_query = x_query.size(0)
        features = torch.cat((context_features, x_query.numerical), dim=0)
        targets = torch.cat(
            (context_targets, context_targets.new_zeros(num_query)), dim=0
        )
        active_features = torch.tensor(
            [features.size(-1)], dtype=torch.long, device=features.device
        )
        output = self.model(
            features.unsqueeze(0),
            targets.unsqueeze(0),
            torch.tensor(
                [num_context], dtype=torch.long, device=features.device
            ),
            categorical_mask.unsqueeze(0),
            active_features,
        )[0, num_context:]

        if classes is None:
            return TableTensor.from_tensor(
                output[..., :1], columns=("prediction",)
            )
        return TableTensor.from_tensor(
            output[..., : len(classes)],
            columns=tuple(str(value) for value in classes.tolist()),
        )

    @staticmethod
    def _table(value: Tensor | TableTensor) -> TableTensor:
        return (
            value
            if isinstance(value, TableTensor)
            else TableTensor.from_tensor(value)
        )

    def _features(self, value: Tensor | TableTensor) -> TableTensor:
        table = self._table(value)
        if table.dim() != 2:
            raise ValueError(
                "TabFM requires unbatched two-dimensional features"
            )
        invalid = table.active_stypes - {
            Stype.numerical,
            Stype.categorical,
            Stype.id,
        }
        if invalid:
            stypes = ", ".join(str(stype) for stype in invalid)
            raise ValueError(f"TabFM does not support feature stypes {stypes}")
        return table

    def _target(self, value: Tensor | TableTensor) -> TableTensor:
        table = self._table(value)
        if table.dim() != 2:
            raise ValueError(
                "TabFM requires unbatched two-dimensional targets"
            )
        if self.task == "classification":
            if table.active_stypes != {Stype.categorical}:
                raise ValueError(
                    "Classification requires a categorical target"
                )
            if bool(table.categorical.code.lt(0).any()):
                raise ValueError("Classification targets must not be missing")
            return table
        assert self.task == "regression"
        if table.active_stypes != {Stype.numerical}:
            raise ValueError("Regression requires a numerical target")
        if table.numerical.is_complex() or not bool(
            table.numerical.isfinite().all()
        ):
            raise ValueError("Regression targets must be finite real values")
        return table

    def _target_data(self, y: TableTensor) -> tuple[Tensor, Tensor | None]:
        if self.task == "classification":
            classes = y.categorical.categories[0]
            if len(classes) > self.model.max_classes:
                raise ValueError(
                    f"TabFM supports at most {self.model.max_classes} classes"
                )
            return y.categorical.code[:, 0], classes
        return y.numerical[:, 0], None

    @staticmethod
    def _validate_transformed(
        table: TableTensor, raw_schema: TableSchema
    ) -> None:
        if table.active_stypes - {Stype.numerical, Stype.id}:
            raise ValueError(
                "TabFM recipes must produce only numerical features and IDs"
            )
        source = set(raw_schema.columns[Stype.numerical]) | set(
            raw_schema.columns[Stype.categorical]
        )
        if not set(table.columns[Stype.numerical]) <= source:
            raise ValueError(
                "TabFM recipes must preserve original feature names; IDs "
                "are metadata only"
            )
