"""Compile TableTensor output conversion with empty text storage."""

import json

import torch

from sdm.tensor import TableTensor, VarLenTensor


def _convert_table(x):
    return (
        TableTensor.from_tensor(x.to(torch.bfloat16))
        .to(torch.float32)
        .numerical
    )


def _convert_empty(x):
    return x.to(dtype=torch.float32, copy=True)


for fullgraph in (False, True):
    for kind in ("table", "empty_payload"):
        torch._dynamo.reset()
        f = torch.compile(
            _convert_table if kind == "table" else _convert_empty,
            fullgraph=fullgraph,
            dynamic=True,
            backend="inductor",
        )
        try:
            with torch.inference_mode():
                for rows in (4, 9, 0, 3):
                    if kind == "table":
                        x = torch.arange(
                            rows * 3, dtype=torch.float32
                        ).reshape(rows, 3)
                        expected = _convert_table(x)
                        actual = f(x)
                        torch.testing.assert_close(
                            actual, expected, atol=0, rtol=0
                        )
                    else:
                        x = VarLenTensor(
                            data=torch.empty(0, dtype=torch.bfloat16),
                            offset=torch.zeros(rows + 1, dtype=torch.int32),
                            valid=None,
                            size=(rows,),
                        )
                        expected = _convert_empty(x)
                        actual = f(x)
                        assert actual.size() == expected.size()
                        torch.testing.assert_close(
                            actual._data, expected._data
                        )
                        torch.testing.assert_close(
                            actual._offset, expected._offset
                        )
        except Exception as error:  # noqa: BLE001
            print(  # noqa: T201
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "kind": kind,
                        "fullgraph": fullgraph,
                        "status": "fail",
                        "error": str(error),
                    }
                ),
                flush=True,
            )
        else:
            print(  # noqa: T201
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "kind": kind,
                        "fullgraph": fullgraph,
                        "status": "pass",
                    }
                ),
                flush=True,
            )
