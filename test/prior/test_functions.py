# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm.prior import (
    Correlated,
    CorrelatedContext,
    LogUniform,
    LogUniformInt,
    Mixture,
)
from sdm.prior.activations import (
    CallableActivation,
    StandardizedActivation,
    identity,
)
from sdm.prior.functions import (
    ConcatParentMerger,
    MixtureFunction,
    MixtureParentMerger,
    RandomDiscretization,
    RandomEMAssignment,
    RandomFunction,
    RandomGP,
    RandomLinear,
    RandomNN,
    RandomProduct,
    RandomQuadratic,
    RandomTree,
    ReduceParentMerger,
)
from sdm.prior.matrix_samplers import KernelMatrixSampler


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "fn",
    [
        RandomTree(num_trees=5, depth=3),
        RandomEMAssignment(p=1.0, q=1.5, m=4),
        RandomEMAssignment(p=2.0, q=2.0),
        RandomGP(a=3.0, alpha=2.0, num_features=16),
        RandomProduct(MixtureFunction([RandomLinear(), RandomQuadratic()])),
        RandomLinear(),
        RandomQuadratic(max_channels=3),
        RandomNN(
            num_layers=1,
            hidden_channels=7,
            activation=StandardizedActivation(
                CallableActivation(Tensor.tanh), scale=2.0
            ),
        ),
        RandomNN(
            num_layers=3,
            hidden_channels=7,
            activation=StandardizedActivation(
                CallableActivation(Tensor.tanh), scale=2.0
            ),
        ),
        RandomDiscretization(num_centers=3, p=1.5),
        RandomDiscretization(num_centers=100, p=2.0),
        MixtureFunction(
            functions=[RandomLinear(), RandomQuadratic()],
        ),
    ],
    ids=[
        "tree",
        "em-fixed",
        "em-default",
        "gp",
        "product",
        "linear",
        "quadratic",
        "nn-one",
        "nn-three",
        "discrete",
        "capped",
        "mixture",
    ],
)
def test_function_shape_dtype_and_replay(
    fn: RandomFunction, dtype: torch.dtype
) -> None:
    x = torch.randn(13, 5, dtype=dtype)
    generator = torch.Generator()
    state = generator.get_state()
    with CorrelatedContext(generator=generator):
        actual = fn.forward(x=x, out_channels=3, generator=generator)
    generator.set_state(state)
    with CorrelatedContext(generator=generator):
        repeated = fn.forward(x=x, out_channels=3, generator=generator)

    assert actual.shape == (13, 3)
    assert actual.dtype == dtype
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)


def test_discretization_limits_distinct_outputs() -> None:
    fn = RandomDiscretization(num_centers=3, p=2.0)
    actual = fn.forward(x=torch.randn(17, 5), out_channels=4)
    assert actual.unique(dim=0).size(0) <= 3


def test_concat_parent_values() -> None:
    parent_values = [torch.randn(13, 2), torch.randn(13, 5)]
    random_fn = RandomLinear()
    generator = torch.Generator()
    state = generator.get_state()
    actual = ConcatParentMerger().forward(
        parent_values=parent_values,
        random_fn=random_fn,
        out_channels=4,
        generator=generator,
    )
    generator.set_state(state)
    expected = random_fn.forward(
        x=torch.cat(parent_values, dim=-1),
        out_channels=4,
        generator=generator,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class IdentityLinear(RandomLinear):
    def forward(
        self,
        x: torch.Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        assert x.size(-1) == out_channels
        return x


def test_tree_constant_rows_share_leaves() -> None:
    x = torch.ones(17, 3)
    actual = RandomTree(5, 4).forward(x, out_channels=2)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, actual[:1].expand_as(actual))


def test_em_assignments() -> None:
    x = torch.randn(17, 3)
    fn = RandomEMAssignment(
        p=2.0, q=2.0, bandwidth=0.0, m=4, linear=IdentityLinear()
    )
    generator = torch.Generator()
    state = generator.get_state()
    actual = fn.forward(x, out_channels=4, generator=generator)
    generator.set_state(state)
    indices = torch.randint(17, (1, 4), generator=generator)
    centers = x[indices] + torch.randn(1, 4, 3, generator=generator)
    expected = (-torch.cdist(x.unsqueeze(0), centers).square()).softmax(-1)
    torch.testing.assert_close(actual, expected.squeeze(0))
    torch.testing.assert_close(actual.sum(-1), torch.ones(17))


@pytest.mark.parametrize("product_kernel", [False, True])
def test_gp_kernel_branches(product_kernel: bool) -> None:
    x = torch.randn(17, 3)
    fn = RandomGP(a=3.0, alpha=0.0, num_features=16)
    fn.product_kernel = Mixture([product_kernel])
    actual = fn.forward(x, out_channels=2, generator=torch.Generator())
    assert actual.isfinite().all()
    if product_kernel:
        # The independent-coordinate kernel does not use the projection scale.
        fn.alpha = fn.a
        expected = fn.forward(x, out_channels=2, generator=torch.Generator())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        # A zero projection maps every input to the same GP value.
        torch.testing.assert_close(actual, actual[:1].expand_as(actual))


def test_product_function() -> None:
    functions = MixtureFunction([RandomLinear(), RandomQuadratic()])
    x = torch.randn(13, 3)
    generator = torch.Generator()
    state = generator.get_state()
    actual = RandomProduct(functions).forward(
        x, out_channels=2, generator=generator
    )
    generator.set_state(state)
    first = functions.pick(generator=generator)
    second = functions.pick(generator=generator)
    expected = first.forward(x, 2, generator=generator) * second.forward(
        x, 2, generator=generator
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("aggr", ["sum", "prod", "amax", "logsumexp"])
def test_reduce_parents(aggr: str) -> None:
    parents = [torch.randn(13, 2), torch.randn(13, 5)]
    fn = RandomLinear()
    generator = torch.Generator()
    state = generator.get_state()
    actual = ReduceParentMerger(aggr).forward(
        parent_values=parents,
        random_fn=fn,
        out_channels=3,
        generator=generator,
    )
    generator.set_state(state)
    mapped = torch.stack(
        [fn.forward(x, 3, generator=generator) for x in parents]
    )
    expected = getattr(torch, aggr)(mapped, dim=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mixture_parent_merger_weights() -> None:
    parents = [torch.randn(13, 3), torch.randn(13, 3)]
    merger = MixtureParentMerger(
        [
            (ConcatParentMerger(), 0.0),
            (ReduceParentMerger("sum"), 1.0),
        ]
    )
    actual = merger.forward(
        parent_values=parents, random_fn=IdentityLinear(), out_channels=3
    )
    torch.testing.assert_close(actual, parents[0] + parents[1])


def test_nn_with_standardized_activation() -> None:
    fn = RandomNN(
        num_layers=1,
        hidden_channels=3,
        linear=IdentityLinear(),
        activation=StandardizedActivation(CallableActivation(identity)),
    )
    fn.start_activation = Mixture([True])
    fn.end_activation = Mixture([False])
    x = torch.arange(51, dtype=torch.float32).reshape(17, 3)
    actual = fn.forward(x, out_channels=3)
    torch.testing.assert_close(
        actual.mean(0), torch.zeros(3), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        actual.std(0, correction=0), torch.ones(3), atol=1e-3, rtol=0
    )


def test_kernel_matrices_share_one_dataset_across_functions() -> None:
    def make_kernel() -> KernelMatrixSampler:
        return KernelMatrixSampler(
            num_centers=Correlated(LogUniformInt(1, 15), key="num_centers"),
            gamma=Correlated(LogUniform(0.1, 10.0), key="gamma"),
        )

    x = torch.randn(13, 4, dtype=torch.float64)
    generator = torch.Generator()
    functions = [
        RandomLinear(matrix_sampler=make_kernel()),
        RandomQuadratic(matrix_sampler=make_kernel()),
        RandomDiscretization(
            num_centers=3,
            p=2.0,
            linear=RandomLinear(matrix_sampler=make_kernel()),
        ),
        RandomEMAssignment(
            p=2.0,
            q=2.0,
            m=4,
            linear=RandomLinear(matrix_sampler=make_kernel()),
        ),
        RandomNN(
            num_layers=2,
            hidden_channels=7,
            activation=StandardizedActivation(
                CallableActivation(torch.Tensor.tanh)
            ),
            linear=RandomLinear(matrix_sampler=make_kernel()),
        ),
    ]
    with CorrelatedContext(generator=generator):
        for fn, channels in zip(functions, [3, 5, 4, 6, 2]):
            actual = fn.forward(x, out_channels=channels, generator=generator)
            assert actual.shape == (13, channels)
            assert actual.isfinite().all()
