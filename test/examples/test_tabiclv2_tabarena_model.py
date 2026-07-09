from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("autogluon")

from examples.benchmarking import tabiclv2_tabarena_model as adapter
from examples.benchmarking.tabiclv2_tabarena_model import (
    DeviceAllocation,
    SDMTabICLv2Model,
)
from sdm import TableTensor
from sdm.cache import Cache
from sdm.processing import FeaturePermute, MeanImpute, Recipe, Sequential

pytestmark = pytest.mark.tabarena


class _FakeTabICLv2:
    instances: ClassVar[list[_FakeTabICLv2]] = []

    def __init__(self, *, pretrained: bool, device: torch.device) -> None:
        assert pretrained is False
        self.device = torch.device(device)
        self.loaded: tuple[Path, str] | None = None
        self.fit_x: TableTensor | None = None
        self.fit_y: TableTensor | None = None
        self.fit_num_estimators: int | None = None
        self._caches: list[Cache] | None = None
        self.cleared = False
        self.__class__.instances.append(self)

    def load_regression_checkpoint(
        self,
        path: Path,
        sha256: str,
    ) -> _FakeTabICLv2:
        self.loaded = (Path(path), sha256)
        return self

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe(features=[MeanImpute(), FeaturePermute(method="random")])

    def fit(
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        num_estimators: int,
    ) -> None:
        self.fit_x = x
        self.fit_y = y
        self.fit_num_estimators = num_estimators
        cache = Cache({"probe": torch.ones(1, device=self.device)})
        cache.freeze()
        self._caches = [cache]

    def predict(self, x: TableTensor) -> torch.Tensor:
        point = x.numerical.sum(dim=-1)
        return point.unsqueeze(-1).repeat(1, 999)

    def clear(self) -> None:
        self.cleared = True
        self._caches = None

    def to(self, device: torch.device) -> _FakeTabICLv2:
        self.device = torch.device(device)
        return self


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> list[_FakeTabICLv2]:
    _FakeTabICLv2.instances = []
    monkeypatch.setattr(adapter, "TabICLv2", _FakeTabICLv2)
    monkeypatch.setattr(
        adapter,
        "decode_regression_quantiles",
        lambda quantiles, _target: quantiles.mean(dim=-1),
    )
    return _FakeTabICLv2.instances


def _model(
    *,
    problem_type: str = "regression",
    **hyperparameters: object,
) -> SDMTabICLv2Model:
    params: dict[str, object] = {
        "checkpoint_path": "/tmp/tabicl-regressor.ckpt",
        "checkpoint_sha256": "0" * 64,
        "seed": 0,
        "num_estimators": 1,
    }
    params.update(hyperparameters)
    return SDMTabICLv2Model(
        path="",
        name="SDMTabICLv2Model",
        problem_type=problem_type,
        eval_metric=None,
        hyperparameters=params,
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.nan, 4.0],
            "feature_b": [0.5, 1.5, 2.5, 3.5],
        }
    )


def _target() -> pd.Series:
    return pd.Series([0.1, 0.2, 0.3, 0.4], name="target")


def test_regression_contract_is_sdm_owned(
    fake_backend: list[_FakeTabICLv2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()

    def fail_preprocess(*args, **kwargs):
        raise AssertionError("AutoGluon preprocessing must not run")

    monkeypatch.setattr(model, "preprocess", fail_preprocess)
    model.fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    predictions = model.predict(_features())

    assert SDMTabICLv2Model.preprocess_data is False
    assert len(fake_backend) == 1
    assert fake_backend[0].loaded == (
        Path("/tmp/tabicl-regressor.ckpt"),
        "0" * 64,
    )
    assert fake_backend[0].fit_x is not None
    assert not torch.isnan(fake_backend[0].fit_x.numerical).any()
    assert fake_backend[0].fit_num_estimators == 1
    assert len(fake_backend[0]._caches or []) == 1
    assert predictions.shape == (len(_features()),)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, model.predict(_features()))


def test_regression_only_guard(fake_backend: list[_FakeTabICLv2]) -> None:
    model = _model(problem_type="binary")
    with pytest.raises(ValueError, match="regression only"):
        model._fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    assert not fake_backend


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame({"category": ["a", "b"]}),
        pd.DataFrame({"is_active": [True, False]}),
        pd.DataFrame({"when": pd.to_datetime(["2025-01-01", "2025-01-02"])}),
        pd.DataFrame({"customer_id": [1, 2]}),
    ],
)
def test_non_numerical_feature_schema_is_rejected(
    fake_backend: list[_FakeTabICLv2],
    X: pd.DataFrame,
) -> None:
    model = _model()
    with pytest.raises(ValueError, match="only numerical"):
        model._fit(X=X, y=pd.Series([0.0, 1.0]), num_cpus=1, num_gpus=0)
    assert not fake_backend


def test_prediction_requires_fit_and_same_feature_schema(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    model = _model()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(_features())

    model.fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    with pytest.raises(ValueError, match="order differs"):
        model.predict(_features()[["feature_b", "feature_a"]])
    with pytest.raises(ValueError, match="missing"):
        model.predict(pd.DataFrame({"feature_a": [1.0]}))
    with pytest.raises(ValueError, match="unexpected"):
        model.predict(
            pd.DataFrame(
                {
                    "feature_a": [1.0],
                    "feature_b": [2.0],
                    "extra": [3.0],
                }
            )
        )
    assert len(fake_backend) == 1


def test_repeated_fit_replaces_cached_context(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    model = _model()
    model.fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    first = fake_backend[0]

    next_features = _features().rename(
        columns={"feature_a": "new_a", "feature_b": "new_b"}
    )
    model.fit(X=next_features, y=_target(), num_cpus=1, num_gpus=0)

    assert len(fake_backend) == 2
    assert first.cleared is True
    assert model.predict(next_features).shape == (len(next_features),)
    with pytest.raises(ValueError, match="missing"):
        model.predict(_features())


def _fitted_permutation(model: SDMTabICLv2Model) -> torch.Tensor:
    assert model._recipe is not None
    assert isinstance(model._recipe.features, Sequential)
    step = model._recipe.features.steps[-1]
    assert isinstance(step, FeaturePermute)
    return step.permutation.clone()


def test_seeded_fit_is_deterministic_and_restores_rng(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    torch.manual_seed(1234)
    initial_state = torch.random.get_rng_state().clone()
    with adapter._fork_seed(7, torch.device("cpu")):
        torch.rand(1)
    assert torch.equal(torch.random.get_rng_state(), initial_state)

    first = _model(seed=7)
    first.fit(
        X=_features(),
        y=_target(),
        num_cpus=1,
        num_gpus=0,
    )
    first_permutation = _fitted_permutation(first)

    second = _model(seed=7)
    second.fit(
        X=_features(),
        y=_target(),
        num_cpus=1,
        num_gpus=0,
    )
    torch.testing.assert_close(_fitted_permutation(second), first_permutation)

    permutations = {tuple(first_permutation.tolist())}
    for seed in range(8):
        candidate = _model(seed=seed)
        candidate.fit(
            X=_features(),
            y=_target(),
            num_cpus=1,
            num_gpus=0,
        )
        permutations.add(tuple(_fitted_permutation(candidate).tolist()))
    assert len(permutations) > 1


@pytest.mark.parametrize(
    ("hyperparameters", "match"),
    [
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**63}, "seed"),
        ({"num_estimators": True}, "exactly one estimator"),
        ({"num_estimators": 2}, "exactly one estimator"),
    ],
)
def test_invalid_reproducibility_config_is_rejected(
    fake_backend: list[_FakeTabICLv2],
    hyperparameters: dict[str, Any],
    match: str,
) -> None:
    model = _model(**hyperparameters)
    with pytest.raises(ValueError, match=match):
        model.fit(
            X=_features(),
            y=_target(),
            num_cpus=1,
            num_gpus=0,
        )
    assert not fake_backend


def test_device_hook_moves_all_caches_and_preserves_replay_mode(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    model = _model()
    model.fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    backend = fake_backend[0]
    assert backend._caches is not None
    old_caches = list(backend._caches)
    assert all(cache.is_replaying for cache in old_caches)

    model._set_device("cpu")

    assert backend._caches is not None
    assert all(new is not old for new, old in zip(backend._caches, old_caches))
    assert all(cache.is_replaying for cache in backend._caches)
    assert all(cache.device.type == "cpu" for cache in backend._caches)
    assert model.get_device() == "cpu"


def test_device_allocation_rejects_unimplemented_multi_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert DeviceAllocation.from_num_gpus(0).device.type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert DeviceAllocation.from_num_gpus(1).device == torch.device("cuda:0")
    with pytest.raises(NotImplementedError, match="Multi-GPU"):
        DeviceAllocation.from_num_gpus(2)


def test_worker_can_import_model_module() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from examples.benchmarking.tabiclv2_tabarena_model "
            "import SDMTabICLv2Model; print(SDMTabICLv2Model.__name__)",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SDMTabICLv2Model"


def test_runner_seed_validation_matches_adapter_contract() -> None:
    from examples.benchmarking.run_tabiclv2_tabarena_smoke import (
        MAX_SEED,
        _validate_seed,
    )

    assert _validate_seed(0) == 0
    assert _validate_seed(MAX_SEED) == MAX_SEED

    invalid_seeds: tuple[Any, ...] = (True, -1, 2**63, 1.5, "0")
    for invalid_seed in invalid_seeds:
        with pytest.raises(ValueError, match="--seed"):
            _validate_seed(invalid_seed)


def test_tabarena_outer_smoke_construction_disables_preprocessing() -> None:
    pytest.importorskip("tabarena")
    from examples.benchmarking.run_tabiclv2_tabarena_smoke import (
        build_smoke_experiments,
    )

    experiments = build_smoke_experiments(
        checkpoint_path=Path("/tmp/tabicl-regressor.ckpt"),
        checkpoint_sha256="0" * 64,
        seed=0,
        num_cpus=1,
        num_gpus=0,
    )

    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment.method_kwargs["preprocess_data"] is False
    assert experiment.method_kwargs["fit_kwargs"]["num_gpus"] == 0
    assert (
        experiment.method_kwargs["hyperparameters"]["checkpoint_sha256"]
        == "0" * 64
    )
    assert experiment.method_kwargs["hyperparameters"]["seed"] == 0
    assert experiment.method_kwargs["hyperparameters"]["num_estimators"] == 1
