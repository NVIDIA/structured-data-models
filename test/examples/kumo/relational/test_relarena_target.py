"""Public target and native recipe inversion retain the V11 median."""

import pandas as pd
import torch
from examples.kumo.relational._relarena.target import TargetQuantile

import sdm
import sdm.processing as sp
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable


def test_quantile_option_preserves_native_classification_recipe() -> None:
    labels = sdm.TableTensor.from_pandas(
        pd.DataFrame({"label": [False, True, False, True]}),
        stypes={"label": "categorical"},
    )
    features = sdm.TableTensor.from_tensor(torch.arange(4.0).unsqueeze(-1))
    executions = []
    fitted = []
    for quantile in (False, True):
        recipe = sdm.models.KumoRelational.default_recipe()
        if quantile:
            recipe.prepend_target(sp.StypeDispatch(numerical=TargetQuantile()))
        execution = RecipeExecution(recipe)
        members = execution.fit_transform(
            x=EnsembleTable(features, num_members=8),
            y=EnsembleTable(labels, num_members=8),
            related_tables=None,
            generator=torch.Generator().manual_seed(0),
        )
        executions.append(execution)
        fitted.append(members)
    # Native classification reconstructs class logits from fitted categories;
    # it never invokes target inversion (ShuffleCategories is not invertible).
    for standard, quantile in zip(*fitted, strict=True):
        pd.testing.assert_frame_equal(
            standard.y.to_pandas(), quantile.y.to_pandas()
        )
    logits = [
        sdm.TableTensor.from_tensor(
            torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * (index + 1),
            columns=["False", "True"],
        )
        for index in range(8)
    ]
    standard, quantile = [
        execution.transform_output(logits) for execution in executions
    ]
    assert standard.columns == quantile.columns
    torch.testing.assert_close(standard.numerical, quantile.numerical)


def test_v11_endpoints_and_inverse_median() -> None:
    target = sdm.TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    )
    mapping = TargetQuantile().fit(target)
    transformed = mapping.transform(target)
    normal = torch.distributions.Normal(0.0, 1.0)
    epsilon = torch.tensor(1e-7 - torch.finfo(torch.float64).eps)
    torch.testing.assert_close(
        transformed.numerical[[0, -1], 0],
        torch.stack((normal.icdf(epsilon), normal.icdf(1 - epsilon))),
    )
    predictions = sdm.TableTensor.from_tensor(
        transformed.numerical.expand(-1, 3),
        columns=["q001", "q500", "q999"],
    )
    actual = mapping.inverse_transform(predictions)
    assert actual.column_names == {"q500"}
    torch.testing.assert_close(actual.numerical, target.numerical)


def test_native_regression_recipe_inverts_each_member_before_reduction() -> (
    None
):
    contexts = [
        sdm.TableTensor.from_tensor(
            torch.arange(1.0, 9.0).unsqueeze(-1) * (member + 1)
        )
        for member in range(8)
    ]
    recipe = sdm.models.KumoRelational.default_recipe()
    recipe.prepend_target(sp.StypeDispatch(numerical=TargetQuantile()))
    execution = RecipeExecution(recipe)
    members = execution.fit_transform(
        x=EnsembleTable(contexts[0], num_members=8),
        y=EnsembleTable.from_tables(contexts, range(8)),
        related_tables=None,
    )
    outputs = [
        sdm.TableTensor.from_tensor(
            member.y.numerical.expand(-1, 3),
            columns=["q001", "q500", "q999"],
        )
        for member in members
    ]
    restored = execution.inverse_transform_target(outputs)
    for actual, expected in zip(restored, contexts, strict=True):
        assert actual.column_names == {"q500"}
        torch.testing.assert_close(actual.numerical, expected.numerical)
    result = execution.transform_output(restored)
    torch.testing.assert_close(
        result.numerical,
        torch.stack([context.numerical for context in contexts]).mean(0),
    )
