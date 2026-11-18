# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import Stype, TableTensor, Task
from sdm.prior import RandomCauchyDAGSampler
from sdm.prior.converters import (
    DistributionalMixtureConverter,
    IdentityConverter,
    KumaraswamyNumericalConverter,
    MixtureConverter,
    NearestNeighborConverter,
    SoftmaxCategoricalConverter,
)
from sdm.prior.functions import ConcatParentMerger, RandomLinear
from sdm.prior.root_samplers import (
    BallRootSampler,
    MixtureRootSampler,
    NormalRootSampler,
    UniformRootSampler,
)
from sdm.prior.tabular import SchemaSampler, SCMPrior
from sdm.prior.tabular.transform import (
    Identity,
    MaskRandom,
    PriorTransform,
    Relabel,
    ScaleTarget,
)


def _make_prior(task_type: Task, cat_ratio: float) -> SCMPrior:
    return SCMPrior(
        schema_sampler=SchemaSampler(
            task_type=task_type,
            num_columns=4,
            cat_ratio=cat_ratio,
            max_cardinality=3,
            num_classes=3 if task_type == Task.classification else None,
        ),
        dag_sampler=RandomCauchyDAGSampler(num_nodes=1),
        latent_channels=2,
        root_sampler=MixtureRootSampler(
            [NormalRootSampler(), UniformRootSampler(), BallRootSampler()],
        ),
        random_fn=RandomLinear(),
        parent_merger=ConcatParentMerger(),
        numerical_converter=IdentityConverter(),
        categorical_converter=NearestNeighborConverter(p=2.0),
    )


@pytest.mark.parametrize("task_type", [Task.classification, Task.regression])
@pytest.mark.parametrize("cat_ratio", [0.0, 0.5, 1.0])
def test_scm_sample(task_type: Task, cat_ratio: float) -> None:
    prior = _make_prior(task_type, cat_ratio)

    sample = prior.sample(num_rows=64, generator=torch.Generator())

    assert sample is not None
    features, target = sample
    assert features.shape == (64, 4)
    assert target.shape == (64, 1)
    assert features.column_names == {f"x_{i}" for i in range(4)}
    assert target.column_names == {"target"}
    assert features.categorical.size(-1) == round(4 * cat_ratio)
    assert features.numerical.isfinite().all()
    for codes, categories in zip(
        features.categorical.code.unbind(-1), features.categorical.categories
    ):
        assert ((codes >= 0) & (codes < categories.numel())).all()

    if task_type == Task.classification:
        assert target.active_stypes == {Stype.categorical}
        labels = target.categorical.code
        assert ((labels >= 0) & (labels < 3)).all()
        torch.testing.assert_close(
            target.categorical.categories[0], torch.arange(3)
        )
    else:
        assert target.active_stypes == {Stype.numerical}
        assert target.numerical.isfinite().all()


def test_scm_applies_transforms() -> None:
    prior = _make_prior(Task.regression, cat_ratio=0.0)
    prior.transforms = (ScaleTarget(),)

    sample = prior.sample(num_rows=64, generator=torch.Generator())

    assert sample is not None
    torch.testing.assert_close(sample[1].numerical.std(), torch.tensor(1.0))


def test_scm_rejects_nonfinite_output() -> None:
    prior = _make_prior(Task.regression, cat_ratio=0.0)
    prior.root_sampler = NormalRootSampler(mean=torch.nan)

    assert prior.sample(num_rows=64, generator=torch.Generator()) is None


@pytest.mark.parametrize("value", [torch.nan, torch.inf])
@pytest.mark.parametrize("relabel", [False, True])
def test_scm_checks_nonfinite_target_after_transforms(
    monkeypatch: pytest.MonkeyPatch, value: float, relabel: bool
) -> None:
    prior = _make_prior(Task.regression, cat_ratio=1.0)
    converter = IdentityConverter()
    prior.numerical_converter = converter
    forward = converter.forward

    def nonfinite_target(
        slices: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target, feedback = forward(slices=slices, generator=generator)
        return torch.full_like(target, value), feedback

    monkeypatch.setattr(prior.numerical_converter, "forward", nonfinite_target)
    prior.transforms = (Relabel(RandomLinear()) if relabel else Identity(),)

    sample = prior.sample(num_rows=64)

    if not relabel:
        assert sample is None
        return
    assert sample is not None
    assert sample[1].numerical.isfinite().all()


@pytest.mark.parametrize("task_type", [Task.regression, Task.classification])
def test_scm_allows_missing_features_after_transforms(task_type: Task) -> None:
    prior = _make_prior(task_type, cat_ratio=0.0)
    prior.transforms = (MaskRandom(rate=1.0),)

    sample = prior.sample(num_rows=64)

    assert sample is not None
    assert sample[0].numerical.isnan().all()
    if task_type == Task.regression:
        assert sample[1].numerical.isfinite().all()
    else:
        assert (sample[1].categorical.code >= 0).all()


@pytest.mark.parametrize("task_type", [Task.regression, Task.classification])
def test_scm_rejects_nonfinite_category_codes(
    monkeypatch: pytest.MonkeyPatch, task_type: Task
) -> None:
    prior = _make_prior(task_type, cat_ratio=1.0)
    converter = NearestNeighborConverter(p=2.0)
    prior.categorical_converter = converter
    forward = converter.forward

    def nonfinite_codes(
        slices: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes, feedback = forward(slices=slices, generator=generator)
        return torch.full_like(codes, torch.nan, dtype=torch.float), feedback

    monkeypatch.setattr(
        prior.categorical_converter, "forward", nonfinite_codes
    )
    prior.transforms = (ScaleTarget(),)

    assert prior.sample(num_rows=64) is None


@pytest.mark.parametrize("mode", ["center", "function", "embedding"])
def test_scm_converter_feedback(mode: str) -> None:
    prior = _make_prior(Task.classification, cat_ratio=0.5)
    # Mixtures retain the dependencies supplied to their converters.
    prior = SCMPrior(
        schema_sampler=prior.schema_sampler,
        dag_sampler=RandomCauchyDAGSampler(num_nodes=8),
        latent_channels=2,
        root_sampler=prior.root_sampler,
        random_fn=prior.random_fn,
        parent_merger=prior.parent_merger,
        numerical_converter=MixtureConverter(
            [
                DistributionalMixtureConverter(dist_num_conv_prob=1),
                KumaraswamyNumericalConverter(a=1.0, b=2.0),
            ]
        ),
        categorical_converter=MixtureConverter(
            [
                SoftmaxCategoricalConverter(
                    output_mode=mode,
                    root_sampler=prior.root_sampler,
                    random_fn=prior.random_fn,
                )
                if mode == "embedding"
                else NearestNeighborConverter(
                    p=1.5, output_mode=mode, random_fn=prior.random_fn
                ),
            ]
        ),
    )
    actual = prior.sample(64, generator=torch.Generator())
    replay = prior.sample(64, generator=torch.Generator())
    assert actual is not None
    assert replay is not None
    for table, repeated in zip(actual, replay):
        torch.testing.assert_close(table.numerical, repeated.numerical)
        torch.testing.assert_close(
            table.categorical.code, repeated.categorical.code
        )


class _Reject(PriorTransform):
    def __call__(
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        return None


def test_scm_transform_can_reject_task() -> None:
    prior = _make_prior(Task.regression, cat_ratio=0.5)
    prior.transforms = (_Reject(), ScaleTarget())

    assert prior.sample(num_rows=64, generator=torch.Generator()) is None
