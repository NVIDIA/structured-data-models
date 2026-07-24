"""Unit tests for the SDM-native official-policy TabArena control."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.tabiclv2_tabarena.official_policy import (
    FixedCategoryCodePermute,
    FixedFeaturePermute,
    OfficialV2EnsemblePlan,
)
from examples.tabiclv2_tabarena.run_official_policy import config_from_args
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor


def _feature_table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        columns=("x0", "x1", "x2"),
    )


def _target_table() -> TableTensor:
    return TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [2], [-1]], dtype=torch.int64),
            categories=(StringTensor.from_list(["a", "b", "c"]),),
        ),
    )


def test_official_policy_plan_is_fixed_and_paired() -> None:
    plan = OfficialV2EnsemblePlan.build(
        feature_count=5,
        class_count=3,
        random_state=42,
    )

    assert len(plan.members) == 8
    assert [member.normalization for member in plan.members].count("none") == 4
    assert [member.normalization for member in plan.members].count(
        "power"
    ) == 4
    for none_member, power_member in zip(
        plan.members[::2], plan.members[1::2], strict=True
    ):
        assert none_member.normalization == "none"
        assert power_member.normalization == "power"
        assert (
            none_member.feature_permutation == power_member.feature_permutation
        )
        assert none_member.class_permutation == power_member.class_permutation

    assert plan == OfficialV2EnsemblePlan.build(
        feature_count=5,
        class_count=3,
        random_state=42,
    )
    assert plan != OfficialV2EnsemblePlan.build(
        feature_count=5,
        class_count=3,
        random_state=7,
    )
    assert all(
        sorted(member.feature_permutation) == list(range(5))
        for member in plan.members
    )
    assert all(
        member.class_permutation is not None
        and sorted(member.class_permutation) == [0, 1, 2]
        for member in plan.members
    )


def test_official_policy_regression_omits_class_permutations() -> None:
    plan = OfficialV2EnsemblePlan.build(
        feature_count=4,
        class_count=None,
    )

    assert len(plan.members) == 8
    assert all(member.class_permutation is None for member in plan.members)
    with pytest.raises(ValueError, match="at least four distinct"):
        OfficialV2EnsemblePlan.build(feature_count=3, class_count=None)


def test_fixed_feature_permutation_is_bijective_and_invertible() -> None:
    table = _feature_table()
    processor = FixedFeaturePermute((2, 0, 1))

    transformed = processor.fit_transform(table)

    assert transformed.columns[Stype.numerical] == ("x2", "x0", "x1")
    assert torch.equal(
        processor.inverse_transform(transformed).numerical,
        table.numerical,
    )


def test_fixed_class_permutation_preserves_missing_codes() -> None:
    target = _target_table()
    transformed = FixedCategoryCodePermute((2, 0, 1)).fit_transform(target)

    assert transformed.categorical.as_tensor().tolist() == [
        [2],
        [0],
        [1],
        [-1],
    ]
    assert transformed.categorical.categories[0].tolist() == ["b", "c", "a"]
    assert transformed.categorical.tolist() == target.categorical.tolist()


@pytest.mark.parametrize(
    ("processor", "table"),
    [
        (FixedFeaturePermute((0, 1)), _feature_table()),
        (FixedFeaturePermute((0, 0, 1)), _feature_table()),
        (FixedFeaturePermute((0, 1, 3)), _feature_table()),
        (FixedCategoryCodePermute((0, 1)), _target_table()),
        (FixedCategoryCodePermute((0, 0, 1)), _target_table()),
        (FixedCategoryCodePermute((0, 1, 3)), _target_table()),
    ],
)
def test_fixed_permutations_reject_invalid_bijections(
    processor: FixedFeaturePermute | FixedCategoryCodePermute,
    table: TableTensor,
) -> None:
    with pytest.raises(ValueError, match="permutation"):
        processor.fit(table)


def test_official_policy_runner_uses_fixed_eight_member_contract(
    tmp_path: Path,
) -> None:
    config = config_from_args(
        argparse.Namespace(
            output_root=tmp_path / "run",
            num_cpus=1,
            num_gpus=0,
            subset=["lite"],
            datasets=["example"],
            random_state=42,
        )
    )

    assert config.output_root == (tmp_path / "run").resolve()
    assert config.random_state == 42
    assert not hasattr(config, "num_estimators")


def test_official_policy_adapter_fits_eight_member_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.tabiclv2_tabarena.sdm_system as sdm_system
    from sdm.models import TabICLv2

    class FakeTabICLv2:
        default_recipe = TabICLv2.default_recipe

        def __init__(self, *, device: torch.device) -> None:
            self.device = device
            self._caches = None

        def fit(self, **kwargs: object) -> None:
            self._caches = [{"recipe": kwargs["recipe"]}]

    monkeypatch.setattr(sdm_system, "TabICLv2", FakeTabICLv2)
    system = object.__new__(sdm_system.SDMTabICLv2OfficialPolicySystem)
    system._device = torch.device("cpu")
    system.random_state = 42
    features = TableTensor.from_tensor(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [3.0, 8.0, 9.0, 12.0],
            ]
        ),
        columns=("x0", "x1", "x2", "x3"),
    )

    system._fit_tabicl(x=features, y=_target_table())

    assert len(system.model._caches) == 8
    assert len(system.ensemble_plan.members) == 8
    assert [
        member.normalization for member in system.ensemble_plan.members
    ] == [
        "none",
        "power",
    ] * 4


def test_optional_tabicl_policy_parity() -> None:
    preprocessing = pytest.importorskip("tabicl.sklearn.preprocessing")
    generator = preprocessing.EnsembleGenerator(
        classification=True,
        n_estimators=8,
        random_state=42,
    )
    generator.fit(
        torch.arange(50, dtype=torch.float64).reshape(10, 5).numpy(),
        torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]).numpy(),
    )
    plan = OfficialV2EnsemblePlan.build(feature_count=5, class_count=3)
    expected = {
        (
            member.normalization,
            member.feature_permutation,
            member.class_permutation,
        )
        for member in plan.members
    }
    observed = {
        (normalization, tuple(feature), tuple(classes))
        for normalization, configs in generator.ensemble_configs_.items()
        for feature, classes in configs
    }
    assert observed == expected


@pytest.mark.skipif(
    os.environ.get("SDM_RUN_TABARENA_SMOKE") != "1",
    reason="Set SDM_RUN_TABARENA_SMOKE=1 to run the real TabArena smoke test",
)
def test_real_tabarena_official_policy_smoke(tmp_path: Path) -> None:
    """Run binary, multiclass, and regression jobs through the new system."""
    pytest.importorskip("autogluon.core.models")
    pytest.importorskip("tabarena")
    from examples.tabiclv2_tabarena.run_official_policy import (
        OfficialPolicyRunConfig,
        run,
    )

    output_root = tmp_path / "tabarena-official-policy-smoke"
    run(
        OfficialPolicyRunConfig(
            output_root=output_root,
            num_cpus=1,
            num_gpus=1 if torch.cuda.is_available() else 0,
            subset=["lite"],
            datasets=[
                "blood-transfusion-service-center",
                "anneal",
                "QSAR_fish_toxicity",
            ],
            random_state=42,
        )
    )

    assert (output_root / "report" / "results_per_split.csv").is_file()
