"""Compile TableTensor output conversion with empty text storage."""

import json

import torch

from sdm.tensor import StringTensor, TableTensor, VarLenTensor


def _convert_table(x):
    return (
        TableTensor.from_tensor(x.to(torch.bfloat16))
        .to(torch.float32)
        .numerical
    )


def _convert_empty(x):
    return x.to(
        dtype=torch.uint8 if isinstance(x, StringTensor) else torch.float32,
        copy=True,
    )


for fullgraph in (False, True):
    for kind in ("table", "empty_payload", "empty_strings"):
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
                        cls = (
                            StringTensor
                            if kind == "empty_strings"
                            else VarLenTensor
                        )
                        dtype = (
                            torch.uint8
                            if kind == "empty_strings"
                            else torch.bfloat16
                        )
                        x = cls(
                            data=torch.empty(0, dtype=dtype),
                            offset=torch.zeros(rows + 1, dtype=torch.int32),
                            valid=torch.arange(rows).remainder(2) == 0,
                            size=(rows,),
                        )
                        expected = _convert_empty(x)
                        actual = f(x)
                        assert actual.size() == expected.size()
                        torch.testing.assert_close(
                            actual._valid, expected._valid
                        )
                        for attr in ("_data", "_offset", "_valid"):
                            assert not torch._C._is_alias_of(
                                getattr(x, attr), getattr(actual, attr)
                            )
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
