from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from torch import Tensor

from sdm import CategoricalTensor, ColumnarTensor, Recipe, Stype, TableTensor
from sdm.models import TabFM
from sdm.models.tabfm import model as model_module
from sdm.models.tabfm.core import _TabFM
from sdm.testing import withCUDA

_Task = Literal["classification", "regression"]


def _core(task: _Task, device: torch.device | str = "cpu") -> _TabFM:
    return _TabFM(
        embed_dim=2,
        max_classes=10,
        col_num_blocks=1,
        col_num_heads=1,
        col_num_inducing_points=2,
        row_num_blocks=1,
        row_num_heads=1,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_num_heads=2,
        feedforward_factor=2,
        feature_group_size=2,
        num_frequencies=2,
        decoder_hidden_channels=5,
        is_classifier=task == "classification",
        device=device,
        dtype=torch.float32,
    )


def _model(
    monkeypatch: pytest.MonkeyPatch,
    task: _Task,
    *,
    core: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> TabFM:
    core = _core(task, device) if core is None else core
    monkeypatch.setattr(
        model_module,
        "_load_tabfm_v1_0_0_from_huggingface",
        lambda **_: core,
    )
    return TabFM(task, device=device, dtype=torch.float32)


def _tables(
    device: torch.device | str = "cpu",
) -> tuple[TableTensor, TableTensor, TableTensor, TableTensor]:
    context = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.categorical: ("kind",),
            Stype.id: ("row_id",),
        },
        numerical=torch.arange(6, device=device, dtype=torch.float32)[:, None],
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0], [1], [0], [1]], device=device),
            categories=(torch.tensor([20, 10], device=device),),
        ),
        id=ColumnarTensor((torch.arange(6, device=device),)),
    )
    query = TableTensor(
        columns={
            "numerical": ("value",),
            "categorical": ("kind",),
            "id": ("row_id",),
        },
        numerical=torch.tensor([[6.0], [7.0]], device=device),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], device=device),
            categories=(torch.tensor([10, 99], device=device),),
        ),
        id=ColumnarTensor((torch.tensor([10, 11], device=device),)),
    )
    classification = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [2], [0], [1], [2]], device=device),
            categories=(torch.tensor([30, 10, 20], device=device),),
        ),
    )
    regression = TableTensor.from_tensor(
        torch.arange(6, device=device, dtype=torch.float32)[:, None],
        columns=("target",),
    )
    return context, classification, regression, query


@withCUDA
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_default_recipe_matches_record_and_replay(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    task: _Task,
) -> None:
    context, classification, regression, query = _tables(device)
    target = classification if task == "classification" else regression
    model = _model(monkeypatch, task, device=device)

    output = model(
        context,
        target,
        query,
        num_estimators=2,
        generator=torch.Generator(device=device).manual_seed(4),
    )
    model.fit(
        context,
        target,
        num_estimators=2,
        generator=torch.Generator(device=device).manual_seed(4),
    )
    replay = model.predict(query)

    assert output.size() == (2, 3 if task == "classification" else 1)
    assert output.columns[Stype.numerical] == (
        ("10", "20", "30") if task == "classification" else ("prediction",)
    )
    assert output.numerical.isfinite().all()
    torch.testing.assert_close(replay.numerical, output.numerical)
    assert replay.columns == output.columns


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_zero_feature_tables_are_supported(
    monkeypatch: pytest.MonkeyPatch,
    task: _Task,
) -> None:
    _, classification, regression, _ = _tables()
    target = classification if task == "classification" else regression
    model = _model(monkeypatch, task)
    context = TableTensor(size=(6,))
    query = TableTensor(size=(2,))
    output = model(
        context,
        target,
        query,
        num_estimators=2,
        generator=torch.Generator().manual_seed(2),
    )
    model.fit(
        context,
        target,
        num_estimators=2,
        generator=torch.Generator().manual_seed(2),
    )
    replay = model.predict(query)
    assert output.size(-2) == 2
    assert output.numerical.isfinite().all()
    torch.testing.assert_close(replay.numerical, output.numerical)


class _MaskCore(torch.nn.Module):
    max_classes = 10

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0))

    def forward(
        self,
        features: Tensor,
        targets: Tensor,
        context_size: Tensor,
        categorical_mask: Tensor,
        active_features: Tensor,
    ) -> Tensor:
        del targets, context_size, active_features
        output = features.new_zeros((*features.shape[:2], self.max_classes))
        output[..., 0] = categorical_mask.sum(dim=-1, keepdim=True)
        return output


def test_categorical_provenance_reaches_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, target, _, query = _tables()
    model = _model(monkeypatch, "classification", core=_MaskCore())
    categorical = model(context, target, query)

    numerical_context = TableTensor.from_tensor(
        torch.cat((context.categorical.code.float(), context.numerical), -1),
        columns=("kind", "value"),
    )
    numerical_query = TableTensor.from_tensor(
        torch.cat((query.categorical.code.float(), query.numerical), -1),
        columns=("kind", "value"),
    )
    numerical = model(numerical_context, target, numerical_query)
    assert (categorical.numerical[:, 0] > numerical.numerical[:, 0]).all()


def test_raw_schema_is_checked_before_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch, "regression")
    context = TableTensor.from_tensor(torch.arange(6.0)[:, None])
    query = TableTensor.from_tensor(torch.tensor([[0], [1]]))
    target = TableTensor.from_tensor(torch.arange(6.0)[:, None])

    with pytest.raises(ValueError, match="same schema"):
        model(context, target, query)
    model.fit(context, target)
    with pytest.raises(ValueError, match="same schema"):
        model.predict(query)


def test_task_target_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    context, classification, regression, query = _tables()
    classifier = _model(monkeypatch, "classification")
    regressor = _model(monkeypatch, "regression")

    with pytest.raises(ValueError, match="categorical target"):
        classifier(context, regression, query)
    missing = classification.replace_blocks(
        categorical=CategoricalTensor(
            code=classification.categorical.code.clone().index_fill_(
                0, torch.tensor([0]), -1
            ),
            categories=classification.categorical.categories,
        )
    )
    with pytest.raises(ValueError, match="must not be missing"):
        classifier(context, missing, query)
    with pytest.raises(ValueError, match="numerical target"):
        regressor(context, classification, query)
    for values in (
        torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [torch.nan]]),
        torch.arange(6)[:, None].to(torch.complex64) * (1 + 1j),
    ):
        invalid = TableTensor(
            columns={Stype.numerical: ("target",)}, numerical=values
        )
        with pytest.raises(ValueError, match="finite real"):
            regressor(context, invalid, query)


def test_class_and_recipe_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = _model(monkeypatch, "classification", core=_MaskCore())
    context = TableTensor.from_tensor(torch.arange(11.0)[:, None])
    query = TableTensor.from_tensor(torch.tensor([[11.0]]))
    target = TableTensor.from_tensor(torch.arange(11)[:, None])
    with pytest.raises(ValueError, match="at most 10 classes"):
        classifier(context, target, query)

    mixed, small_target, _, mixed_query = _tables()
    with pytest.raises(ValueError, match="only numerical features"):
        classifier(mixed, small_target, mixed_query, recipe=Recipe())
    numerical_context = mixed.select_stypes(Stype.numerical)
    numerical_query = mixed_query.select_stypes(Stype.numerical)
    output = classifier(
        numerical_context,
        small_target,
        numerical_query,
        recipe=Recipe(),
        num_estimators=2,
    )
    assert output.size() == (2, 2, 3)
    assert output.columns[Stype.numerical] == ("30", "10", "20")
    assert output.numerical.count_nonzero() == 0

    with pytest.raises(ValueError, match="unbatched"):
        classifier(torch.randn(2, 3, 2), small_target, mixed_query)


def test_constructor_routes_checkpoint_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier = _core("classification")
    hub_kwargs: dict[str, Any] = {}

    def load_hub(**kwargs: Any) -> _TabFM:
        hub_kwargs.update(kwargs)
        return classifier

    monkeypatch.setattr(
        model_module, "_load_tabfm_v1_0_0_from_huggingface", load_hub
    )
    TabFM(
        "classification",
        cache_dir=tmp_path,
        local_files_only=True,
        device="cpu",
        dtype=torch.float32,
    )
    assert hub_kwargs == {
        "task": "classification",
        "cache_dir": tmp_path,
        "local_files_only": True,
        "device": "cpu",
        "dtype": torch.float32,
    }

    regressor = _core("regression")
    local_call: dict[str, Any] = {}

    def load_local(path: str | Path, **kwargs: Any) -> _TabFM:
        local_call.update(path=path, **kwargs)
        return regressor

    monkeypatch.setattr(
        model_module,
        "_load_tabfm_v1_0_0",
        load_local,
    )
    checkpoint_path = tmp_path / "model.safetensors"
    TabFM(
        "regression",
        checkpoint_path=checkpoint_path,
        device="cpu",
        dtype=torch.float32,
    )
    assert local_call == {
        "path": checkpoint_path,
        "task": "regression",
        "device": "cpu",
        "dtype": torch.float32,
    }
    with pytest.raises(ValueError, match="only apply to Hub"):
        TabFM(
            "regression",
            checkpoint_path=checkpoint_path,
            cache_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="only apply to Hub"):
        TabFM(
            "regression",
            checkpoint_path=checkpoint_path,
            local_files_only=True,
        )
