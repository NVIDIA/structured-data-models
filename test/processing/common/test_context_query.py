import torch

import sdm.processing as sp
from sdm import TableTensor
from sdm.processing.execution import RecipeExecution


def test_context_query_dispatch() -> None:
    x = TableTensor.from_tensor(torch.zeros(4, 1))

    recipe = sp.Recipe(
        features=[
            sp.ContextQueryDispatch(
                context=sp.SelectRows(3, method="round_robin"),
                query=sp.Identity(),
            ),
            sp.Standardize(),
        ]
    )
    execution = RecipeExecution(recipe)
    outs = execution.fit_transform(x, x, related_tables=None, num_members=4)
    assert len(outs) == 4
    assert all(out.x.size() == (3, 1) for out in outs)

    outs = execution.transform(x, related_tables=None)
    assert len(outs) == 4
    assert all(out.x.size() == (4, 1) for out in outs)
