"""Compare compiled fitting with eager on synthetic or real data."""

import argparse
import json

import torch

from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import AlignCategories


def _check(
    tables: list[TableTensor], *, sort_by: str, fullgraph: bool
) -> None:
    torch._dynamo.reset()
    processor = AlignCategories(sort_by=sort_by, min_frequency=2)
    compiled = torch.compile(
        processor.fit_transform, fullgraph=fullgraph, dynamic=True
    )
    for table in tables:
        oracle = AlignCategories(sort_by=sort_by, min_frequency=2)
        expected = oracle.fit_transform(table)
        actual = compiled(table)
        torch.testing.assert_close(
            actual.categorical.code, expected.categorical.code, rtol=0, atol=0
        )
        for got, want in zip(
            actual.categorical.categories, expected.categorical.categories
        ):
            assert got.tolist() == want.tolist()
        torch.testing.assert_close(
            processor.transform(table).categorical.code,
            oracle.transform(table).categorical.code,
            rtol=0,
            atol=0,
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="Existing RelBench driver bundle")
    args = parser.parse_args()
    cases = {}
    if args.bundle:
        bundle = torch.load(args.bundle, weights_only=False)
        table = bundle["arms"][0]["related_context"].tables["drivers"]
        table = table.select_stypes("categorical")
        cases["relbench_drivers"] = [table[:rows] for rows in (32, 16, 8)]
    else:
        for name, categories in (
            ("numeric", torch.tensor([30, 10, 20, 40])),
            (
                "string_int64",
                StringTensor.from_list(["red", "blue", "green", "unused"]),
            ),
        ):
            cases[name] = [
                TableTensor(
                    categorical=CategoricalTensor(
                        code=torch.tensor(codes, dtype=torch.int32).view(
                            -1, 1
                        ),
                        categories=(categories,),
                    )
                )
                for codes in (
                    [0, 1, 1, -1],
                    [2, 0, 2, 2, -1, 0],
                    [1, 1, 1],
                    [-1, -1],
                )
            ]
    for case, tables in cases.items():
        for fullgraph in (False, True):
            for sort_by in ("code", "frequency", "value"):
                result = dict(  # noqa: C408
                    version=torch.__version__,
                    case=case,
                    fullgraph=fullgraph,
                    sort_by=sort_by,
                )
                try:
                    _check(tables, sort_by=sort_by, fullgraph=fullgraph)
                    result["status"] = "pass"
                except Exception as error:  # noqa: BLE001
                    result.update(status="fail", error=str(error))
                print(json.dumps(result), flush=True)  # noqa: T201


if __name__ == "__main__":
    _main()
