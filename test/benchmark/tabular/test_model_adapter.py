from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import S_TEXT_EMBEDDING

import sdm
from benchmark.tabular.system import (
    MODEL_CONFIGS,
    ModelConfig,
    ModelFactory,
    SDMModel,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache = None
        self.classes: list[int] = []
        self.num_estimators: int | None = None

    def fit(
        self,
        *,
        y: sdm.TableTensor,
        num_estimators: int | None,
        **_: object,
    ) -> None:
        self.num_estimators = num_estimators
        if y.categorical.size(-1) > 0:
            self.classes = [
                int(value) for value in y.categorical.categories[0].tolist()
            ]
        else:
            self.classes = []

    def predict(self, x: sdm.TableTensor) -> sdm.TableTensor:
        if self.classes:
            values = torch.ones(
                (x.size(-2), len(self.classes)),
                device=x.device,
            )
            values /= values.size(-1)
            columns = [str(value) for value in self.classes]
        else:
            values = torch.ones((x.size(-2), 2), device=x.device)
            columns = ["q1", "q2"]
        return sdm.TableTensor(
            columns={sdm.Stype.numerical: columns},
            numerical=values,
        )

    def clear(self) -> None:
        self._cache = None


def test_abstract_adapter_supports_both_tasks_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    instances: list[_Model] = []

    def factory(task: str, device: torch.device) -> _Model:
        del task
        model = _Model().to(device)
        instances.append(model)
        return model

    monkeypatch.setitem(
        MODEL_CONFIGS,
        "fake",
        ModelConfig(
            name="Fake",
            factory=cast(ModelFactory, factory),
            num_estimators=8,
            autocast_dtype=torch.float16,
        ),
    )
    features = pd.DataFrame(
        {
            "value": np.arange(8, dtype=np.float32),
            "embedding": np.linspace(0.0, 1.0, 8, dtype=np.float32),
        }
    )
    metadata = FeatureMetadata(
        type_map_raw={"value": "float", "embedding": "float"},
        type_group_map_special={S_TEXT_EMBEDDING: ["embedding"]},
    )

    for problem_type, target, num_classes in (
        ("binary", pd.Series([0, 1] * 4, name="target"), 2),
        (
            "regression",
            pd.Series(np.arange(8, dtype=np.float32), name="target"),
            None,
        ),
    ):
        adapter = SDMModel(
            path=str(tmp_path),
            problem_type=problem_type,
            hyperparameters={"model": "fake", "random_state": 7},
        )
        adapter.fit(
            X=features,
            y=target,
            feature_metadata=metadata,
            num_classes=num_classes,
            num_cpus=1,
            num_gpus=0,
        )
        prediction = adapter.predict_proba(features.iloc[:3])

        assert prediction.shape[0] == 3
        assert adapter.text_embedding_columns == ("embedding",)
        assert instances[-1].num_estimators == 8

    assert len(instances) == 2
