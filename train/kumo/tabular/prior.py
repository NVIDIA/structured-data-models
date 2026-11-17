# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import torch.nn.functional as F
from torch import Tensor

import sdm.prior.tabular.transform as pt
from sdm import Task
from sdm.prior import (
    ActivationMatrixSampler,
    BallRootSampler,
    CallableActivation,
    Choice,
    ConcatParentMerger,
    Correlated,
    CorrelatedRootSampler,
    DistributionalMixtureConverter,
    GaussianMatrixSampler,
    IntegerPowActivation,
    InversePowActivation,
    KernelMatrixSampler,
    LogUniform,
    LogUniformInt,
    MixtureActivation,
    MixtureConverter,
    MixtureFunction,
    MixtureMatrixSampler,
    MixtureParentMerger,
    MixtureRootSampler,
    NearestNeighborConverter,
    NormalRootSampler,
    PowerLawMatrixSampler,
    RandomCauchyDAGSampler,
    RandomDiscretization,
    RandomEMAssignment,
    RandomGP,
    RandomLinear,
    RandomNN,
    RandomProduct,
    RandomQuadratic,
    RandomTree,
    ReduceParentMerger,
    ReLUPowActivation,
    SignPowActivation,
    SingularValuesMatrixSampler,
    SoftmaxCategoricalConverter,
    StandardizedActivation,
    Uniform,
    UniformInt,
    UniformRootSampler,
    argsort,
    gaussian,
    heaviside,
    identity,
    log_abs,
    modulo,
    one_hot_argmax,
    rank,
    softmax,
    unit_indicator,
)
from sdm.prior.tabular import SchemaSampler, SCMPrior


def make_prior(
    task_type: Task,
    *,
    correlated_block_prob: float = 0.3,
    relabel_prob: float = 0.1,
    coarsen_prob: float = 0.1,
    high_cardinality_prob: float = 0.05,
    y_heavy_tail_prob: float = 0.2,
    structured_missingness_prob: float = 0.2,
    mcar_missingness_prob: float = 0.01,
    near_empty_missingness_prob: float = 0.05,
    latent_driver_missingness_prob: float = 0.03,
    softmax_argmax_prob: float = 0.15,
) -> SCMPrior:
    """Build the prior used to train Kumo Tabular models.

    Args:
        task_type: Classification or regression task.
        correlated_block_prob: Probability of correlated feature blocks.
        relabel_prob: Probability of replacing the target.
        coarsen_prob: Probability of coarsening features.
        high_cardinality_prob: Probability of high-cardinality recoding.
        y_heavy_tail_prob: Probability of regression target warping.
        structured_missingness_prob: Structured masking probability.
        mcar_missingness_prob: Probability of 1-8% random cell masking.
        near_empty_missingness_prob: Probability of masking column groups
            to few observed rows.
        latent_driver_missingness_prob: Probability of masking driven by
            an unobserved row factor.
        softmax_argmax_prob: Probability of a softmax-converter call using
            bounded-bias argmax.
    """
    schema_sampler = SchemaSampler(
        task_type=task_type,
        num_columns=UniformInt(2, 100),
        cat_ratio=Uniform(-0.5, 1.2),
        max_cardinality=LogUniformInt(2, 200),
        num_classes=UniformInt(2, 10)
        if task_type == Task.classification
        else None,
        independent_cat_ratio=Uniform(-0.1, 1.1),
        max_correlated_cardinality=9,
    )
    dag_sampler = RandomCauchyDAGSampler(num_nodes=LogUniformInt(2, 32))
    latent_channels = Correlated(LogUniformInt(1, 32))
    root_sampler = MixtureRootSampler(
        samplers=[
            (NormalRootSampler(mean=0.0, std=1.0), 0.25),
            (UniformRootSampler(low=-1.0, high=1.0), 0.25),
            (BallRootSampler(), 0.25),
            (
                CorrelatedRootSampler(
                    MixtureRootSampler(
                        samplers=[
                            NormalRootSampler(mean=0.0, std=1.0),
                            UniformRootSampler(low=-1.0, high=1.0),
                        ],
                    )
                ),
                0.25,
            ),
        ],
    )
    fixed_activation = MixtureActivation(
        activations=[
            CallableActivation(function)
            for function in (
                Tensor.tanh,
                F.leaky_relu,
                F.elu,
                identity,
                F.selu,
                F.silu,
                Tensor.relu,
                F.softplus,
                F.relu6,
                F.hardtanh,
                Tensor.sign,
                heaviside,
                gaussian,
                Tensor.exp,
                unit_indicator,
                Tensor.sin,
                Tensor.square,
                Tensor.abs,
                softmax,
                one_hot_argmax,
                argsort,
                F.logsigmoid,
                log_abs,
                rank,
                Tensor.sigmoid,
                Tensor.round,
                modulo,
            )
        ],
        key="TorchActivation.kind",
    )
    node_activation = StandardizedActivation(
        MixtureActivation(
            activations=[
                (fixed_activation, 4.0),
                ReLUPowActivation(exponent=Correlated(LogUniform(0.1, 10.0))),
                SignPowActivation(exponent=Correlated(LogUniform(0.1, 10.0))),
            ],
        ),
        scale=Correlated(LogUniform(1.0, 10.0)),
    )

    matrix_sampler = MixtureMatrixSampler(
        samplers=[
            GaussianMatrixSampler(),
            PowerLawMatrixSampler(),
            SingularValuesMatrixSampler(),
            KernelMatrixSampler(
                num_centers=Correlated(
                    LogUniformInt(1, 15),
                    key="GaussianMixturePoints.num_centers",
                ),
                gamma=Correlated(
                    LogUniform(0.1, 10.0), key="KernelMatrix.scale"
                ),
            ),
            ActivationMatrixSampler(
                matrix_sampler=MixtureMatrixSampler(
                    samplers=[
                        GaussianMatrixSampler(normalize=False),
                        PowerLawMatrixSampler(normalize=False),
                        SingularValuesMatrixSampler(normalize=False),
                        KernelMatrixSampler(
                            num_centers=Correlated(
                                LogUniformInt(1, 15),
                                key="GaussianMixturePoints.num_centers",
                            ),
                            gamma=Correlated(
                                LogUniform(0.1, 10.0),
                                key="KernelMatrix.scale",
                            ),
                            normalize=False,
                        ),
                    ],
                    correlated=True,
                    key="simple_matrix_type",
                ),
                activation=MixtureActivation(
                    activations=[
                        (fixed_activation, 8.0),
                        ReLUPowActivation(
                            exponent=Correlated(LogUniform(0.1, 10.0)),
                        ),
                        SignPowActivation(
                            exponent=Correlated(LogUniform(0.1, 10.0)),
                        ),
                        InversePowActivation(
                            exponent=Correlated(LogUniform(0.1, 10.0)),
                        ),
                        IntegerPowActivation(
                            exponent=Correlated(UniformInt(2, 5)),
                        ),
                    ],
                    key="matrix_activation_type",
                ),
            ),
        ],
        correlated=True,
        key="matrix_type",
    )

    functions = [
        RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        RandomNN(
            num_layers=Correlated(LogUniformInt(1, 3)),
            hidden_channels=Correlated(LogUniformInt(1, 127)),
            activation=node_activation,
            linear=RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        ),
        RandomTree(
            num_trees=Correlated(LogUniformInt(1, 128)),
            depth=Correlated(UniformInt(1, 7)),
        ),
        RandomDiscretization(
            num_centers=Correlated(LogUniformInt(2, 255)),
            p=Correlated(LogUniform(0.5, 4.0)),
            linear=RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        ),
        RandomQuadratic(matrix_sampler=deepcopy(matrix_sampler)),
        RandomEMAssignment(
            p=Choice(1.0, 2.0),
            q=Correlated(Uniform(1.0, 2.0)),
            bandwidth=Correlated(Uniform(0.0, 1.0)),
            linear=RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        ),
        RandomGP(
            a=Correlated(LogUniform(2.0, 20.0)),
            alpha=Correlated(LogUniform(0.5, 10.0)),
            num_features=256,
        ),
    ]
    random_fn = MixtureFunction(
        [
            *functions,
            RandomProduct(MixtureFunction(functions)),
        ]
    )
    parent_merger = MixtureParentMerger(
        [
            ConcatParentMerger(),
            ReduceParentMerger(Choice("sum", "prod", "amax", "logsumexp")),
        ]
    )
    numerical_converter = DistributionalMixtureConverter()
    categorical_converter = MixtureConverter(
        converters=[
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="identity",
            ),
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="index",
            ),
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="center",
            ),
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="function",
                random_fn=random_fn,
            ),
            SoftmaxCategoricalConverter(
                output_mode="identity",
                sample_categories=True,
                argmax_prob=softmax_argmax_prob,
            ),
            SoftmaxCategoricalConverter(
                output_mode="index",
                sample_categories=True,
                argmax_prob=softmax_argmax_prob,
            ),
            SoftmaxCategoricalConverter(
                output_mode="embedding",
                sample_categories=True,
                argmax_prob=softmax_argmax_prob,
                root_sampler=root_sampler,
                random_fn=random_fn,
            ),
        ],
    )
    transforms = (
        pt.Choice(
            pt.CorrelateColumns(),
            pt.Identity(),
            weights=(correlated_block_prob, 1 - correlated_block_prob),
        ),
        pt.ScaleFeatures(),
        pt.ShuffleColumns(),
        pt.Choice(
            pt.Relabel(
                MixtureFunction(
                    [
                        (functions[0], 0.5),
                        (
                            RandomNN(
                                num_layers=Correlated(LogUniformInt(1, 2)),
                                hidden_channels=Correlated(
                                    LogUniformInt(1, 32)
                                ),
                                activation=node_activation,
                                linear=RandomLinear(
                                    matrix_sampler=deepcopy(matrix_sampler)
                                ),
                            ),
                            0.3,
                        ),
                        (functions[-1], 0.2),
                    ]
                )
            ),
            pt.Identity(),
            weights=(relabel_prob, 1 - relabel_prob),
        ),
        pt.Choice(
            pt.Coarsen(categorical=True, standardize=True),
            pt.Identity(),
            weights=(coarsen_prob, 1 - coarsen_prob),
        ),
        pt.Choice(
            pt.RecodeHighCardinality(categorical=True, standardize=True),
            pt.Identity(),
            weights=(high_cardinality_prob, 1 - high_cardinality_prob),
        ),
        pt.Choice(
            pt.WarpTargetTails(),
            pt.ScaleTarget(),
            weights=(y_heavy_tail_prob, 1 - y_heavy_tail_prob),
        )
        if task_type == Task.regression
        else pt.ShuffleClasses(),
        pt.RejectNonfinite(),
        pt.Choice(
            pt.MaskNearEmpty(),
            pt.Choice(
                pt.MaskStructured(),
                pt.Identity(),
                weights=(
                    structured_missingness_prob,
                    1 - structured_missingness_prob,
                ),
            ),
            weights=(
                near_empty_missingness_prob,
                1 - near_empty_missingness_prob,
            ),
        ),
        pt.Choice(
            pt.MaskRandom(Uniform(0.01, 0.08)),
            pt.Identity(),
            weights=(mcar_missingness_prob, 1 - mcar_missingness_prob),
        ),
        pt.Choice(
            pt.MaskLatent(),
            pt.Identity(),
            weights=(
                latent_driver_missingness_prob,
                1 - latent_driver_missingness_prob,
            ),
        ),
    )
    return SCMPrior(
        schema_sampler=schema_sampler,
        dag_sampler=dag_sampler,
        latent_channels=latent_channels,
        root_sampler=root_sampler,
        random_fn=random_fn,
        parent_merger=parent_merger,
        numerical_converter=numerical_converter,
        categorical_converter=categorical_converter,
        transforms=transforms,
    )
