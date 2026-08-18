import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import TabFM
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable


def test_recipe_prepares_mixed_features_without_query_leakage() -> None:
    context = TableTensor(
        columns={
            Stype.numerical: ("value", "constant"),
            Stype.categorical: ("kind",),
        },
        numerical=torch.tensor(
            [[0.0, 1.0], [1.0, 1.0], [torch.nan, 1.0], [3.0, 1.0]]
        ),
        categorical=CategoricalTensor(
            code=torch.tensor([[1], [-1], [1], [2]]),
            categories=(torch.tensor([10, 20, 30]),),
        ),
    )
    query = TableTensor(
        columns={
            Stype.numerical: ("value", "constant"),
            Stype.categorical: ("kind",),
        },
        numerical=torch.tensor([[1000.0, 2.0], [torch.nan, 2.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]]),
            categories=(torch.tensor([20, 99]),),
        ),
    )
    processor = TabFM.default_recipe().features

    transformed = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=1),
        generator=torch.Generator().manual_seed(0),
    ).table(0)
    transformed_query = processor.transform_ensemble(
        EnsembleTable(query, num_members=1)
    ).table(0)

    assert set(transformed.columns[Stype.numerical]) == {"kind", "value"}
    assert transformed.numerical.isfinite().all()
    assert transformed_query.numerical.isfinite().all()
    kind = transformed_query.columns[Stype.numerical].index("kind")
    assert (
        transformed_query.numerical[0, kind]
        != transformed_query.numerical[1, kind]
    )
    torch.testing.assert_close(
        transformed_query.numerical[1, kind],
        transformed.numerical[1, kind],
    )


def test_recipe_randomly_selects_at_most_500_features() -> None:
    table = TableTensor.from_tensor(torch.arange(1800.0).view(3, 600))
    selected: list[set[str]] = []
    for seed in (0, 1):
        output = TabFM.default_recipe().features.fit_transform_ensemble(
            EnsembleTable(table, num_members=1),
            generator=torch.Generator().manual_seed(seed),
        )
        columns = output.table(0).columns[Stype.numerical]
        assert len(columns) == 500
        selected.append(set(columns))

    assert selected[0] != selected[1]


def test_recipe_ensembles_feature_views_and_restores_class_order() -> None:
    features = TableTensor.from_tensor(
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [4.0, 8.0],
                [8.0, 16.0],
                [16.0, 32.0],
            ]
        ),
        columns=("x0", "x1"),
    )
    target = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [2], [0], [1], [2]]),
            categories=(torch.tensor([30, 10, 20]),),
        ),
    )
    execution = RecipeExecution(TabFM.default_recipe())
    contexts = execution.fit_transform(
        x=features,
        y=target,
        related_tables=None,
        num_members=8,
        generator=torch.Generator().manual_seed(0),
    )

    assert (
        len({context.x.columns[Stype.numerical] for context in contexts}) > 1
    )
    assert (
        len(
            {
                tuple(context.y.categorical.categories[0].tolist())
                for context in contexts
            }
        )
        > 1
    )
    even = [context.x.numerical for context in contexts[::2]]
    odd = [context.x.numerical for context in contexts[1::2]]
    assert all(
        torch.equal(value.sort(-1).values, even[0].sort(-1).values)
        for value in even
    )
    assert all(
        torch.equal(value.sort(-1).values, odd[0].sort(-1).values)
        for value in odd
    )
    assert not torch.allclose(even[0].sort(-1).values, odd[0].sort(-1).values)

    logits = torch.arange(24.0).view(8, 3)
    outputs = []
    for member, context in enumerate(contexts):
        classes = context.y.categorical.categories[0].tolist()
        by_class = dict(zip((10, 20, 30), logits[member], strict=True))
        outputs.append(
            TableTensor.from_tensor(
                torch.stack([by_class[value] for value in classes]).view(1, 3),
                columns=tuple(str(value) for value in classes),
            )
        )

    output = execution.transform_output(outputs)
    output_classes = tuple(
        int(value) for value in output.columns[Stype.numerical]
    )
    canonical = dict(zip((10, 20, 30), logits.mean(0), strict=True))
    expected = torch.stack([canonical[value] for value in output_classes])
    expected = expected.view(1, 3).div(0.9).softmax(-1)
    torch.testing.assert_close(output.numerical, expected)


def test_recipe_restores_target_scale_and_reduces_estimators() -> None:
    features = TableTensor.from_tensor(torch.arange(8.0).view(4, 2))
    target = TableTensor.from_tensor(
        torch.tensor([[10.0], [20.0], [40.0], [80.0]]),
        columns=("target",),
    )
    execution = RecipeExecution(TabFM.default_recipe())
    execution.fit_transform(
        x=features,
        y=target,
        related_tables=None,
        num_members=3,
        generator=torch.Generator().manual_seed(3),
    )
    scaled = torch.tensor([[[-1.0]], [[0.0]], [[1.0]]])
    outputs = [
        TableTensor.from_tensor(member, columns=("prediction",))
        for member in scaled
    ]

    restored = execution.inverse_transform_target(outputs)
    output = execution.transform_output(restored)

    expected = target.numerical.mean(0, keepdim=True)
    torch.testing.assert_close(output.numerical, expected)
