import argparse, traceback, json
import torch
from sdm import TableTensor
from sdm.models.kumo.tabular.recipe import default_recipe as tabular
from sdm.models.kumo.relational.recipe import default_recipe as relational

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["tabular", "relational"], default="tabular")
p.add_argument("--mode", default="construct")
p.add_argument("--backend", default="eager")
p.add_argument("--fullgraph", action="store_true")
a = p.parse_args()
factory = {"tabular": tabular, "relational": relational}[a.model]
x = torch.arange(60, dtype=torch.float32).reshape(20, 3).sin()
table = TableTensor.from_tensor(x)
if a.mode == "construct":

    def fn(x):
        recipe = factory()
        return x + int(recipe.features.requires_fit)

    args = (x,)
elif a.mode == "transform":
    recipe = factory()
    recipe.features.fit(table)

    def fn(table):
        return recipe.features.transform(table).numerical

    args = (table,)
else:
    recipe = factory()

    def fn(table):
        return recipe.features.fit_transform(table).numerical

    args = (table,)
try:
    with torch.inference_mode():
        compiled = torch.compile(fn, backend=a.backend, fullgraph=a.fullgraph)
        out = compiled(*args)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "model": a.model,
                "mode": a.mode,
                "fullgraph": a.fullgraph,
                "backend": a.backend,
                "result": "PASS",
                "shape": list(out.shape),
            }
        )
    )
except Exception as e:
    traceback.print_exc()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "model": a.model,
                "mode": a.mode,
                "fullgraph": a.fullgraph,
                "backend": a.backend,
                "result": "FAIL",
                "error": str(e).splitlines()[0],
            }
        )
    )
