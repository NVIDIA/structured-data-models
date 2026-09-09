"""Low-rank PCA preserves logical members and fixed-width related routes."""

import pytest
import torch
from examples.kumo.relational._relarena.text import ContextPCA

import sdm
import sdm.processing as sp
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable


def test_nonempty_context_signs_padding_and_missing_query() -> None:
    context = sdm.TableTensor.from_tensor(
        torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    )
    processor = ContextPCA(4).fit(context)
    query = sdm.TableTensor.from_tensor(
        torch.tensor([[0.0, 0.0], [2.0, 3.0], [3.0, 4.0]])
    )
    output = processor.transform(query).numerical
    torch.testing.assert_close(output[:2], torch.zeros(2, 4))
    torch.testing.assert_close(output[2, 0], torch.tensor(2.0).sqrt())
    torch.testing.assert_close(output[:, 2:], torch.zeros(3, 2))


def test_all_empty_context_cannot_fit() -> None:
    context = sdm.TableTensor.from_tensor(torch.zeros(3, 2))
    with pytest.raises(RuntimeError, match="no nonempty"):
        ContextPCA(1).fit(context)


def test_context_cap_uses_fixed_cpu_selection_not_caller_generator() -> None:
    values = torch.arange(1.0, 50_003.0).unsqueeze(-1)
    context = sdm.TableTensor.from_tensor(values)
    selected = torch.randperm(
        len(values), generator=torch.Generator().manual_seed(0)
    )[:50_000]
    query = sdm.TableTensor.from_tensor(torch.tensor([[60_000.0]]))
    processor = ContextPCA(1).fit(
        context, generator=torch.Generator().manual_seed(713)
    )
    torch.testing.assert_close(
        processor.transform(query).numerical,
        query.numerical - values[selected].mean(),
    )


@pytest.mark.parametrize("components", [32, 384])
def test_heterogeneous_pca_rank_preserves_eight_related_members(
    components: int,
) -> None:
    dimension = components + 8
    columns = {
        sdm.Stype.numerical: [f"embedding_{i}" for i in range(dimension)]
    }
    rows = [2, 3, 4, 5, 6, 8, 16, components + 2]
    contexts = {
        name: [
            sdm.TableTensor(
                columns=columns,
                numerical=torch.randn(count, dimension),
            )
            for count in counts
        ]
        for name, counts in {"first": rows, "second": rows[::-1]}.items()
    }
    queries = {
        name: sdm.TableTensor(
            columns=columns,
            numerical=torch.randn(3, dimension),
        )
        for name in contexts
    }
    x = sdm.TableTensor.from_tensor(torch.ones(3, 1), columns=["value"])
    y = sdm.TableTensor.from_tensor(torch.arange(3.0).unsqueeze(-1))
    execution = RecipeExecution(
        sp.Recipe(features=sp.TableDispatch(related=ContextPCA(components)))
    )
    fitted = execution.fit_transform(
        x=EnsembleTable(x, num_members=8),
        y=EnsembleTable(y, num_members=8),
        related_tables=sdm.RelatedTables(
            relationships=[],
            task_links=[],
            tables={
                name: EnsembleTable.from_tables(tables, range(8))
                for name, tables in contexts.items()
            },
        ),
    )
    output = execution.transform(
        x=EnsembleTable(x, num_members=8),
        related_tables=sdm.RelatedTables(
            relationships=[],
            task_links=[],
            tables={
                name: EnsembleTable(query, num_members=8)
                for name, query in queries.items()
            },
        ),
    )
    assert len(fitted) == len(output) == 8
    for index, member in enumerate(output):
        assert member.related_tables is not None
        for name, tables in contexts.items():
            context = tables[index]
            expected = (
                ContextPCA(components).fit(context).transform(queries[name])
            )
            actual = member.related_tables.tables[name]
            assert actual.shape == (3, components)
            assert actual.columns == expected.columns
            torch.testing.assert_close(actual.numerical, expected.numerical)
