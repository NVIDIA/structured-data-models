from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, cast

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
from sdm.models import TabICLv2
from sdm.processing import (
    FeaturePermute,
    InvertibleMixin,
    MeanImpute,
    Recipe,
    Sequential,
    StandardScale,
)

pytestmark = pytest.mark.tabarena


class _FakeTabICLv2:
    instances: ClassVar[list[_FakeTabICLv2]] = []

    def __init__(
        self,
        *,
        pretrained: bool,
        device: torch.device,
        batch_size_limit: int | None = None,
    ) -> None:
        assert pretrained is False
        self.device = torch.device(device)
        self.batch_size_limit = batch_size_limit
        self.loaded: tuple[Path, str] | None = None
        self.received_fit_x: TableTensor | None = None
        self.received_fit_y: TableTensor | None = None
        self.received_predict_x: TableTensor | None = None
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
        return TabICLv2.default_recipe()

    def fit(
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        recipe: Recipe,
        num_estimators: int,
        generator: torch.Generator | None,
    ) -> None:
        self.received_fit_x = x
        self.received_fit_y = y
        self.fit_num_estimators = num_estimators
        fitted_recipe = copy.deepcopy(recipe)
        self.fit_x = fitted_recipe.features.fit_transform(
            x,
            generator=generator,
        )
        self.fit_y = fitted_recipe.target.fit_transform(
            y,
            generator=generator,
        )
        cache = Cache(
            {
                "probe": torch.ones(1, device=self.device),
                "recipe": fitted_recipe,
                "classes": None,
            }
        )
        cache.freeze()
        self._caches = [cache]

    def predict(self, x: TableTensor) -> TableTensor:
        assert self._caches is not None
        self.received_predict_x = x
        recipe = self._caches[0]["recipe"]
        assert isinstance(recipe, Recipe)
        model_features = recipe.features.transform(x)
        point = model_features.numerical.sum(dim=-1)
        quantiles = TableTensor.from_tensor(
            point.unsqueeze(-1).repeat(1, 999),
            columns=tuple(f"quantile_{index}" for index in range(999)),
        )
        target = cast(InvertibleMixin, recipe.target)
        restored = target.inverse_transform(quantiles)
        return TableTensor.from_tensor(
            torch.stack([restored.numerical], dim=0),
            columns=tuple(f"quantile_{index}" for index in range(999)),
        )

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


def _mixed_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "segment": ["basic", "premium", "basic", "premium"],
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
    assert fake_backend[0].received_fit_x is not None
    assert torch.isnan(fake_backend[0].received_fit_x.numerical).any()
    assert fake_backend[0].fit_x is not None
    assert not torch.isnan(fake_backend[0].fit_x.numerical).any()
    assert fake_backend[0].fit_num_estimators == 1
    assert (
        fake_backend[0].batch_size_limit
        == adapter.DEFAULT_ATTENTION_BATCH_SIZE_LIMIT
    )
    assert len(fake_backend[0]._caches or []) == 1
    assert predictions.shape == (len(_features()),)
    assert np.isfinite(predictions).all()
    assert fake_backend[0].received_predict_x is not None
    assert torch.isnan(fake_backend[0].received_predict_x.numerical).any()
    np.testing.assert_allclose(predictions, model.predict(_features()))


def test_regression_only_guard(fake_backend: list[_FakeTabICLv2]) -> None:
    model = _model(problem_type="binary")
    with pytest.raises(ValueError, match="regression only"):
        model._fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    assert not fake_backend


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame({"when": pd.to_datetime(["2025-01-01", "2025-01-02"])}),
        pd.DataFrame({"customer_id": [1, 2]}),
    ],
)
def test_unsupported_feature_schema_is_rejected(
    fake_backend: list[_FakeTabICLv2],
    X: pd.DataFrame,
) -> None:
    model = _model()
    with pytest.raises(ValueError, match="only numerical and categorical"):
        model._fit(X=X, y=pd.Series([0.0, 1.0]), num_cpus=1, num_gpus=0)
    assert not fake_backend


def test_mixed_feature_schema_is_train_fitted_and_allows_unseen_categories(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    model = _model()
    model.fit(X=_mixed_features(), y=_target(), num_cpus=1, num_gpus=0)
    predictions = model.predict(
        pd.DataFrame(
            {
                "amount": [1.5, 2.5],
                "segment": ["premium", "unseen"],
            }
        )
    )

    backend = fake_backend[0]
    assert backend.received_fit_x is not None
    assert backend.received_fit_x.categorical.size(-1) == 1
    assert backend.fit_x is not None
    assert backend.fit_x.categorical.size(-1) == 0
    assert backend.received_predict_x is not None
    assert backend.received_predict_x.categorical.size(-1) == 1
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


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


def test_recipe_construction_can_be_overridden(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    replacement = Recipe(features=[MeanImpute()])

    class _RecipeModel(SDMTabICLv2Model):
        def _build_recipe(self, model: TabICLv2) -> Recipe:
            assert model is fake_backend[0]
            return replacement

    model = _RecipeModel(
        path="",
        name="RecipeModel",
        problem_type="regression",
        eval_metric=None,
        hyperparameters={
            "checkpoint_path": "/tmp/tabicl-regressor.ckpt",
            "checkpoint_sha256": "0" * 64,
            "seed": 0,
            "num_estimators": 1,
        },
    )
    model.fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)

    assert model._recipe is not replacement
    assert fake_backend[0]._caches is not None
    assert model._recipe is fake_backend[0]._caches[0]["recipe"]


def test_regression_prediction_is_not_inverse_transformed_twice(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    class _ScaledTargetModel(SDMTabICLv2Model):
        def _build_recipe(self, model: TabICLv2) -> Recipe:
            return Recipe(
                features=[MeanImpute()],
                target=[StandardScale()],
            )

    model = _ScaledTargetModel(
        path="",
        name="ScaledTargetModel",
        problem_type="regression",
        eval_metric=None,
        hyperparameters={
            "checkpoint_path": "/tmp/tabicl-regressor.ckpt",
            "checkpoint_sha256": "0" * 64,
            "seed": 0,
            "num_estimators": 1,
        },
    )
    model.fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    predictions = model.predict(_features())

    backend = fake_backend[0]
    assert backend.received_predict_x is not None
    assert model._recipe is not None
    transformed = model._recipe.features.transform(backend.received_predict_x)
    point = transformed.numerical.sum(dim=-1, keepdim=True)
    target = cast(InvertibleMixin, model._recipe.target)
    expected = target.inverse_transform(
        TableTensor.from_tensor(point, columns=("target",))
    ).numerical.squeeze(-1)
    twice_inverted = target.inverse_transform(
        TableTensor.from_tensor(expected.unsqueeze(-1), columns=("target",))
    ).numerical.squeeze(-1)

    np.testing.assert_allclose(
        predictions,
        expected.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(expected, twice_inverted)


def test_failed_fit_preserves_model_attribute_and_can_recover(
    fake_backend: list[_FakeTabICLv2],
) -> None:
    model = _model()

    with pytest.raises(ValueError, match="only numerical and categorical"):
        model.fit(
            X=pd.DataFrame(
                {"when": pd.to_datetime(["2025-01-01", "2025-01-02"])}
            ),
            y=pd.Series([0.0, 1.0]),
            num_cpus=1,
            num_gpus=0,
        )

    assert hasattr(model, "model")
    assert model.model is None
    assert model._sdm_model is None

    model.fit(X=_features(), y=_target(), num_cpus=1, num_gpus=0)
    assert model.model is fake_backend[0]
    assert model.predict(_features()).shape == (len(_features()),)


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

    first = _model(seed=7)
    first.fit(
        X=_features(),
        y=_target(),
        num_cpus=1,
        num_gpus=0,
    )
    first_permutation = _fitted_permutation(first)
    assert torch.equal(torch.random.get_rng_state(), initial_state)

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
