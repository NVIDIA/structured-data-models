from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

import sdm
from benchmark.tabular.system import SDMSystem


class _Model:
    def __init__(self) -> None:
        self.fit_targets: list[pd.Series] = []
        self.classes: list[object] = []

    def fit(self, *, y: sdm.TableTensor, **_: object) -> None:
        target = y.to_pandas().iloc[:, 0]
        self.fit_targets.append(target)
        self.classes = y.categorical.categories[0].tolist()

    def predict(self, x: sdm.TableTensor) -> sdm.TableTensor:
        probabilities = torch.ones(
            (x.size(-2), len(self.classes)),
            device=x.device,
        )
        probabilities /= probabilities.size(-1)
        return sdm.TableTensor(
            columns={
                sdm.Stype.numerical: [str(label) for label in self.classes]
            },
            numerical=probabilities,
        )

    def clear(self) -> None:
        pass


def _make_system(
    *,
    many_class: bool,
) -> tuple[SDMSystem, _Model]:
    model = _Model()

    def factory(**_: object) -> _Model:
        return model

    system = object.__new__(SDMSystem)
    system._config = SimpleNamespace(
        name="KumoTabular",
        factory=factory,
        num_estimators=8,
        autocast_dtype=torch.float16,
    )
    system._checkpoint = None
    system._max_context_size = None
    system._max_columns = None
    system._max_cells = None
    system._batch_size = 2
    system._many_class = many_class
    system._many_class_codebook = None
    return system, model


def _fit(
    system: SDMSystem,
    labels: np.ndarray,
) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "a": np.arange(len(labels), dtype=np.float32),
            "b": np.arange(len(labels), dtype=np.float32) % 3,
        }
    )
    system._fit_system(
        features,
        pd.Series(labels),
        target_name="target",
        problem_type="multiclass",
        random_state=7,
    )
    return features.iloc[:3]


def test_many_class_adapter_runs_each_code_row() -> None:
    system, model = _make_system(many_class=True)
    query = _fit(system, np.tile(np.arange(100, 111), 2))

    assert system._many_class_codebook is not None
    assert model.fit_targets == []

    probabilities = system._predict_proba(query)

    assert len(model.fit_targets) == len(system._many_class_codebook)
    assert all(target.nunique() <= 10 for target in model.fit_targets)
    assert probabilities.columns.tolist() == list(range(100, 111))
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    assert np.isfinite(probabilities.to_numpy()).all()


def test_many_class_adapter_bypasses_small_targets() -> None:
    system, model = _make_system(many_class=True)
    query = _fit(system, np.tile(np.arange(100, 110), 2))

    assert system._many_class_codebook is None
    assert len(model.fit_targets) == 1

    probabilities = system._predict_proba(query)

    assert probabilities.columns.tolist() == list(range(100, 110))
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )


def test_disabled_many_class_adapter_uses_the_standard_path() -> None:
    system, model = _make_system(many_class=False)
    _fit(system, np.tile(np.arange(100, 111), 2))

    assert system._many_class_codebook is None
    assert len(model.fit_targets) == 1


if __name__ == "__main__":
    tests: tuple[Callable[[], Any], ...] = (
        test_many_class_adapter_runs_each_code_row,
        test_many_class_adapter_bypasses_small_targets,
        test_disabled_many_class_adapter_uses_the_standard_path,
    )
    for test in tests:
        test()
