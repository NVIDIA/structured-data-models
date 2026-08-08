import torch

from sdm import CategoricalTensor, ColumnarTensor, Stype, TableTensor
from sdm.models.tabfm.recipe import _categorical_mask, default_recipe
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


@withCUDA
def test_basic_features_preserve_columns_and_finite_values(
    device: torch.device,
) -> None:
    codes = torch.tensor(([1, 0] * 9) + [2, -1], device=device)[:, None]
    value = torch.tensor(
        [*range(19), 1000.0],
        dtype=torch.float32,
        device=device,
    )[:, None]
    value[5] = torch.nan
    context = TableTensor(
        columns={
            Stype.numerical: ("value", "constant"),
            Stype.categorical: ("kind",),
            Stype.id: ("row_id",),
        },
        numerical=torch.cat((value, torch.full_like(value, 7.0)), dim=1),
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.tensor([20, 10, 30], device=device),),
        ),
        id=ColumnarTensor((torch.arange(20, device=device),)),
    )
    query_value = torch.tensor(
        [[20.0], [torch.nan], [-1000.0], [10.0]],
        device=device,
    )
    query = TableTensor(
        columns={
            Stype.numerical: ("value", "constant"),
            Stype.categorical: ("kind",),
            Stype.id: ("row_id",),
        },
        numerical=torch.cat(
            (query_value, torch.full_like(query_value, 8.0)),
            dim=1,
        ),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [2], [-1]], device=device),
            categories=(torch.tensor([20, 10, 99], device=device),),
        ),
        id=ColumnarTensor((torch.arange(100, 104, device=device),)),
    )
    processor = default_recipe().features
    generator = torch.Generator(device=device).manual_seed(0)

    transformed = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=1),
        generator=generator,
    ).table(0)
    transformed_query = processor.transform_ensemble(
        EnsembleTable(query, num_members=1)
    ).table(0)

    assert transformed.columns[Stype.numerical] == ("kind", "value")
    assert transformed.numerical.isfinite().all()
    assert transformed_query.numerical.isfinite().all()
    assert (
        transformed.columns[Stype.id],
        transformed.id.equal(context.id),
        transformed_query.columns[Stype.id],
        transformed_query.id.equal(query.id),
    ) == (("row_id",), True, ("row_id",), True)
    assert torch.equal(
        _categorical_mask(
            transformed,
            context.columns[Stype.categorical],
        ),
        torch.tensor([True, False], device=device),
    )


@withCUDA
def test_vectorized_recipe_composes_normalization_and_class_undo(
    device: torch.device,
) -> None:
    features = TableTensor.from_tensor(
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [4.0, 8.0],
                [8.0, 16.0],
                [16.0, 32.0],
            ],
            device=device,
        ),
        columns=("x0", "x1"),
    )
    target = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1], [2], [-1], [0], [1]],
                device=device,
            ),
            categories=(torch.tensor([30, 10, 20], device=device),),
        ),
    )
    execution = RecipeExecution(default_recipe())
    contexts = execution.fit_transform(
        x=features,
        y=target,
        related_tables=None,
        num_members=8,
        generator=torch.Generator(device=device).manual_seed(0),
    )
    target_orders = [
        tuple(context.y.categorical.categories[0].tolist())
        for context in contexts
    ]
    feature_orders = [
        context.x.columns[Stype.numerical] for context in contexts
    ]
    assert target_orders[0] == (10, 20, 30)
    assert len(set(target_orders)) == 3
    assert len(set(feature_orders[::2])) > 1
    assert len(set(feature_orders[1::2])) > 1
    assert len(set(target_orders[::2])) > 1
    assert len(set(target_orders[1::2])) > 1

    normalized = []
    for context in contexts:
        columns = context.x.columns[Stype.numerical]
        indices = torch.tensor(
            [columns.index("x0"), columns.index("x1")],
            device=device,
        )
        normalized.append(context.x.numerical.index_select(-1, indices))
    for index in range(2, 8, 2):
        torch.testing.assert_close(normalized[index], normalized[0])
    for index in range(3, 8, 2):
        torch.testing.assert_close(normalized[index], normalized[1])
    assert not torch.allclose(normalized[0], normalized[1])
    expected_none = normalized[0].new_tensor(
        [-0.9411238, -0.7589708, -0.5768178, -0.2125118, 0.5161001, 1.9733241]
    )
    expected_power = normalized[1].new_tensor(
        [-1.3017136, -0.8622983, -0.4736715, 0.1553961, 0.9001185, 1.5821687]
    )
    torch.testing.assert_close(
        normalized[0][:, 0], expected_none, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        normalized[1][:, 0],
        expected_power,
        atol=5e-5,
        rtol=5e-4,
    )

    member = torch.arange(8, dtype=torch.float32, device=device)
    canonical_logits = torch.stack(
        (member, 2 * (member % 3) - 1, -member.square() / 4),
        dim=-1,
    )
    outputs = []
    for index, context in enumerate(contexts):
        classes = context.y.categorical.categories[0].tolist()
        logits = dict(zip((10, 20, 30), canonical_logits[index], strict=True))
        outputs.append(
            TableTensor.from_tensor(
                torch.stack([logits[value] for value in classes]).unsqueeze(0),
                columns=tuple(str(value) for value in classes),
            )
        )
    output = execution.transform_output(outputs)
    expected = canonical_logits.mean(dim=0, keepdim=True).div(0.9).softmax(-1)
    mean_probabilities = canonical_logits.div(0.9).softmax(-1).mean(0)
    assert output.columns[Stype.numerical] == ("10", "20", "30")
    torch.testing.assert_close(output.numerical, expected)
    assert not torch.allclose(expected[0], mean_probabilities)


def test_categorical_mask_follows_feature_cap() -> None:
    numerical_columns = tuple(f"x{index}" for index in range(500))
    features = TableTensor(
        columns={
            Stype.numerical: numerical_columns,
            Stype.categorical: ("kind",),
        },
        numerical=torch.arange(2000.0).view(4, 500),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [0], [1], [1]]),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    output = (
        default_recipe()
        .features.fit_transform_ensemble(
            EnsembleTable(features, num_members=1),
            generator=torch.Generator().manual_seed(0),
        )
        .table(0)
    )
    columns = output.columns[Stype.numerical]

    assert len(columns) == 500
    assert "kind" in columns
    assert _categorical_mask(output, ("kind",)).tolist() == [
        column == "kind" for column in columns
    ]


def test_regression_inverse_precedes_ensemble_mean() -> None:
    features = TableTensor.from_tensor(torch.arange(8.0).view(4, 2))
    target = TableTensor.from_tensor(
        torch.tensor([[10.0], [20.0], [40.0], [80.0]]),
        columns=("target",),
    )
    execution = RecipeExecution(default_recipe())
    execution.fit_transform(
        x=features,
        y=target,
        related_tables=None,
        num_members=3,
        generator=torch.Generator().manual_seed(3),
    )
    scaled = torch.tensor([[[-1.0], [0.0]], [[0.0], [1.0]], [[1.0], [2.0]]])
    outputs = [
        TableTensor.from_tensor(member, columns=("prediction",))
        for member in scaled
    ]

    restored = execution.inverse_transform_target(outputs)
    output = execution.transform_output(restored)

    mean = target.numerical.mean(dim=0)
    scale = target.numerical.var(dim=0, correction=0).sqrt()
    expected = scaled.mean(dim=0) * scale + mean
    torch.testing.assert_close(output.numerical, expected)
