# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.prior import Choice, Constant, LogUniformInt, Mixture
from sdm.prior.converters import (
    CategoricalConverter,
    DistributionalMixtureConverter,
    IdentityConverter,
    KumaraswamyNumericalConverter,
    MixtureConverter,
    NearestNeighborConverter,
    SoftmaxCategoricalConverter,
)
from sdm.prior.functions import RandomFunction
from sdm.prior.root_samplers import NormalRootSampler


@pytest.mark.parametrize("nested", [False, True])
def test_identity_converter(nested: bool) -> None:
    expected = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    slices = expected.unsqueeze(1)
    if nested:
        slices = torch.nested.as_nested_tensor(
            list(slices.unbind()), layout=torch.jagged
        )

    values, feedback = IdentityConverter().forward(slices)

    torch.testing.assert_close(values, expected)
    for actual, original in zip(feedback.unbind(), slices.unbind()):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize("nested", [False, True], ids=["dense", "ragged"])
def test_nearest_neighbor_converter(nested: bool) -> None:
    cardinalities = (2, 3) if nested else (3, 3)
    columns = [
        torch.arange(k * 8, dtype=torch.float32).reshape(k, 8)
        for k in cardinalities
    ]
    slices = (
        torch.nested.as_nested_tensor(columns, layout=torch.jagged)
        if nested
        else torch.stack(columns)
    )
    values, feedback = NearestNeighborConverter(p=2.0).forward(
        slices, generator=torch.Generator()
    )

    generator = torch.Generator()
    use_full_slice = Mixture(
        items=[False, True],
        correlated=True,
        key="CategoricalConverter.use_full_slice",
    )
    expected = []
    for column, k in zip(columns, cardinalities):
        x = column.T
        if not use_full_slice.pick(generator=generator):
            d = LogUniformInt(1, k - 1).sample(generator=generator)
            x = x[..., torch.randperm(k, generator=generator)[:d]]
        rows = torch.rand(8, generator=generator).argsort()[:k]
        centers = x[rows]
        distances = (x[:, None] - centers[None]).square().sum(dim=-1)
        expected.append(distances.argmin(dim=-1))

    torch.testing.assert_close(values, torch.stack(expected))
    for actual, original in zip(feedback.unbind(), columns):
        torch.testing.assert_close(actual, original)


class DoubleFunction(RandomFunction):
    def forward(
        self,
        x: torch.Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return 2 * x


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", ["identity", "index", "center", "function"])
def test_neighbor_feedback(nested: bool, mode: str) -> None:
    columns = [torch.randn(k, 16) for k in ([2, 3] if nested else [3, 3])]
    slices = (
        torch.nested.as_nested_tensor(columns, layout=torch.jagged)
        if nested
        else torch.stack(columns)
    )
    converter = NearestNeighborConverter(
        p=0.7, output_mode=Choice(mode), random_fn=DoubleFunction()
    )
    labels, feedback = converter.forward(slices, generator=torch.Generator())
    centers = NearestNeighborConverter(p=0.7, output_mode="center").forward(
        slices, generator=torch.Generator()
    )[1]
    for c, (column, mod, center) in enumerate(
        zip(columns, feedback.unbind(), centers.unbind())
    ):
        assert ((labels[c] >= 0) & (labels[c] < column.size(0))).all()
        if mode == "identity":
            torch.testing.assert_close(mod, column)
        elif mode == "index":
            torch.testing.assert_close(
                mod, labels[c].to(column.dtype).expand_as(column)
            )
        else:
            torch.testing.assert_close(
                mod, center * (2 if mode == "function" else 1)
            )
            # Every feedback center is an original row, in all channels.
            assert (center.T[:, None] == column.T[None]).all(-1).any(-1).all()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", ["identity", "index", "embedding"])
@pytest.mark.parametrize("argmax_prob", [0.0, 1.0])
def test_softmax_feedback(nested: bool, mode: str, argmax_prob: float) -> None:
    columns = [torch.randn(k, 32) for k in ([2, 4] if nested else [4, 4])]
    slices = (
        torch.nested.as_nested_tensor(columns, layout=torch.jagged)
        if nested
        else torch.stack(columns)
    )
    converter = SoftmaxCategoricalConverter(
        output_mode=mode,
        argmax_prob=argmax_prob,
        root_sampler=NormalRootSampler(),
        random_fn=DoubleFunction(),
    )
    labels, feedback = converter.forward(slices)
    for c, (column, mod) in enumerate(zip(columns, feedback.unbind())):
        assert ((labels[c] >= 0) & (labels[c] < column.size(0))).all()
        assert mod.shape == column.shape
        assert mod.isfinite().all()
        if mode == "identity":
            torch.testing.assert_close(mod, column)
        elif mode == "index":
            torch.testing.assert_close(
                mod, labels[c].to(column.dtype).expand_as(column)
            )
        else:
            for label in labels[c].unique():
                rows = mod[:, labels[c] == label]
                torch.testing.assert_close(rows, rows[:, :1].expand_as(rows))


@pytest.mark.parametrize("nested", [False, True])
def test_kumaraswamy(nested: bool) -> None:
    x = torch.randn(3, 1, 32)
    slices = (
        torch.nested.as_nested_tensor(list(x.unbind()), layout=torch.jagged)
        if nested
        else x
    )
    values, feedback = KumaraswamyNumericalConverter(a=2.0, b=3.0).forward(
        slices
    )
    scaled = (x[:, 0] - x[:, 0].amin(-1, keepdim=True)) / (
        x[:, 0].amax(-1, keepdim=True) - x[:, 0].amin(-1, keepdim=True)
    )
    torch.testing.assert_close(values, 1 - (1 - scaled.square()).pow(3))
    for original, mod in zip(x.unbind(), feedback.unbind()):
        torch.testing.assert_close(original, mod)


def test_distributional_rounding_and_feedback() -> None:
    converter = DistributionalMixtureConverter(
        dist_num_conv_prob=1,
        dist_count_prob=1,
        dist_warp_prob=0,
    )
    converter.return_sampled = Constant(1)
    converter.feedback_gain = Constant(2.0)
    assert set(converter.sample_num_channels(3).tolist()) <= {1, 3}
    x = torch.randn(3, 3, 64)
    values, feedback = converter.forward(x)
    torch.testing.assert_close(values, values.round())
    torch.testing.assert_close(feedback[:, 1:], x[:, 1:])
    std, mean = torch.std_mean(values, dim=-1, correction=0, keepdim=True)
    original_std = x[:, 0].std(dim=-1, correction=0, keepdim=True)
    expected = 2 * (original_std + 1e-4) * (values - mean) / (std + 1e-4)
    torch.testing.assert_close(feedback[:, 0], expected)


@pytest.mark.parametrize("mode", ["id", "kumar"])
def test_distributional_warping(mode: str) -> None:
    converter = DistributionalMixtureConverter(dist_num_conv_prob=0)
    converter.warp_modes = Mixture([mode])
    x = torch.randn(2, 1, 32)
    actual, feedback = converter.forward(x, generator=torch.Generator())
    if mode == "id":
        torch.testing.assert_close(actual, x[:, 0])
    else:
        assert ((actual >= 0) & (actual <= 1)).all()
    torch.testing.assert_close(feedback, x)


def test_numerical_mixture_selection() -> None:
    identity = IdentityConverter()
    kumar = KumaraswamyNumericalConverter(a=2.0, b=3.0)
    mixture = MixtureConverter(
        [identity, MixtureConverter([kumar])], correlated=False
    )
    selected = mixture.sample(32, generator=torch.Generator())
    assert len(selected) == 32
    assert set(selected) == {identity, kumar}
    assert selected == mixture.sample(32, generator=torch.Generator())

    # A new draw does not invalidate the converters selected for another table.
    mixture.sample(7)
    slices = torch.linspace(0, 1, 16).reshape(1, 1, 16)
    for converter in reversed(selected):
        assert converter.sample_num_channels(1).tolist() == [1]
        actual, feedback = converter.forward(slices)
        expected = slices[:, 0]
        if converter is kumar:
            expected = 1 - (1 - expected.square()).pow(3)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(feedback, slices)


def test_numerical_mixture_weights() -> None:
    identity = IdentityConverter()
    distributional = DistributionalMixtureConverter()
    mixture = MixtureConverter(
        [(identity, 0.0), (MixtureConverter([distributional]), 1.0)],
        correlated=False,
    )
    assert mixture.sample(8) == [distributional] * 8
    assert mixture.sample(0) == []


@pytest.mark.parametrize("nested", [False, True])
def test_categorical_mixture_selection(nested: bool) -> None:
    neighbor = NearestNeighborConverter(2.0, "index")
    softmax = SoftmaxCategoricalConverter("index")
    mixture = MixtureConverter(
        [neighbor, MixtureConverter([softmax])], correlated=False
    )
    selected = mixture.sample(32, generator=torch.Generator())
    assert set(selected) == {neighbor, softmax}
    assert selected == mixture.sample(32, generator=torch.Generator())
    mixture.sample(7)
    for converter in selected:
        columns = [torch.randn(k, 32) for k in ([2, 3] if nested else [3, 3])]
        slices = (
            torch.nested.as_nested_tensor(columns, layout=torch.jagged)
            if nested
            else torch.stack(columns)
        )
        labels, feedback = converter.forward(slices)
        for label, mod in zip(labels, feedback.unbind()):
            torch.testing.assert_close(mod, label.to(mod.dtype).expand_as(mod))
            assert ((label >= 0) & (label < mod.size(0))).all()


def test_categorical_mixture_weights() -> None:
    neighbor = NearestNeighborConverter(2.0)
    softmax = SoftmaxCategoricalConverter("identity")
    mixture = MixtureConverter(
        [(neighbor, 0.0), (MixtureConverter([softmax]), 1.0)],
        correlated=False,
    )
    assert mixture.sample(8) == [softmax] * 8
    assert mixture.sample(0) == []


@pytest.mark.parametrize(
    ("converter", "dependency"),
    [
        (
            NearestNeighborConverter(p=2.0, output_mode=Choice("function")),
            "random_fn",
        ),
        (
            SoftmaxCategoricalConverter(
                output_mode=Choice("embedding"), random_fn=DoubleFunction()
            ),
            "root_sampler",
        ),
        (
            SoftmaxCategoricalConverter(
                output_mode=Choice("embedding"),
                root_sampler=NormalRootSampler(),
            ),
            "random_fn",
        ),
    ],
)
def test_feedback_requires_dependency(
    converter: CategoricalConverter, dependency: str
) -> None:
    slices = torch.randn(1, 2, 8)
    with pytest.raises(AssertionError, match=f"requires {dependency}"):
        converter.forward(slices)
