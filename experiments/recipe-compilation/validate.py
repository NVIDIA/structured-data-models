"""Validate cached default-recipe transforms on real numerical data."""

import argparse
import json
import traceback

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models.kumo.relational.recipe import default_recipe as relational
from sdm.models.kumo.tabular.recipe import default_recipe as tabular
from sdm.processing.execution import RecipeExecution

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["tabular", "relational"], required=True)
p.add_argument("--data", default=None)
p.add_argument("--categorical", action="store_true")
p.add_argument("--rows", type=int, nargs="+", default=[31, 47, 31])
p.add_argument("--members", type=int, default=1)
p.add_argument("--fullgraph", action="store_true")
p.add_argument("--dynamic", action="store_true")
a = p.parse_args()
result = dict(
    torch=torch.__version__,
    model=a.model,
    members=a.members,
    fullgraph=a.fullgraph,
    dynamic=a.dynamic,
    categorical=a.categorical,
    backend="inductor",
    data="sklearn breast_cancer; 300 context rows; constant column appended",
)
try:
    if a.data:
        x, y = torch.load(a.data, weights_only=True)
    else:
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        x = torch.tensor(data.data, dtype=torch.float32)
        y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(-1)
    x = torch.cat((x, x.new_ones((len(x), 1))), dim=-1)

    def make_table(tensor):
        if not a.categorical:
            return TableTensor.from_tensor(tensor)
        return TableTensor(
            numerical=tensor[:, 3:],
            categorical=CategoricalTensor(
                code=tensor[:, :3].round().long() % 3,
                categories=(torch.arange(3),) * 3,
            ),
        )

    execution = RecipeExecution(
        {"tabular": tabular, "relational": relational}[a.model]()
    )
    with torch.inference_mode():
        execution.fit_transform(
            x=make_table(x[:300]),
            y=TableTensor.from_tensor(y[:300]),
            related_tables=None,
            num_members=a.members,
        )
        before = {
            name: value.clone()
            for name, value in execution.recipe.features.named_buffers()
        }

        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(
            execution.transform,
            backend="inductor",
            fullgraph=a.fullgraph,
            dynamic=a.dynamic,
        )
        errors = []
        shapes = []
        for n in a.rows:
            table = make_table(x[300 : 300 + n])
            expected = execution.transform(table, None)
            actual = compiled(table, None)
            for ref_member, out_member in zip(expected, actual, strict=True):
                assert out_member.x.columns == ref_member.x.columns
                ref, out = ref_member.x.numerical, out_member.x.numerical
                torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
                errors.append(float((out - ref).abs().max()))
                shapes.append(list(out.shape))
        for name, value in execution.recipe.features.named_buffers():
            torch.testing.assert_close(
                value, before[name], rtol=0, atol=0, equal_nan=True
            )
    result.update(
        status="PASS",
        max_abs=max(errors),
        shapes=shapes,
        fitted_buffers_unchanged=True,
        compile_stats=dict(torch._dynamo.utils.counters["stats"]),
    )
except Exception as e:
    traceback.print_exc()
    result.update(status="FAIL", error=str(e).splitlines()[0])
print(json.dumps(result))
