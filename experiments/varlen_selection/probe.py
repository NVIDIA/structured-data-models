import json, torch, traceback
from sdm import StringTensor


def gather(text, scores):
    return text.index_select(0, scores.nonzero().flatten())


for dtype in (torch.int32, torch.int64):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        compiled = torch.compile(gather, fullgraph=fullgraph, dynamic=True)
        try:
            for values, scores in (
                (["red", "blue", "green", "unused"], [0, 1, 1, 0]),
                (["", "red", None, "longer string", "z"], [1, 0, 1, 1, 1]),
                (["a", "bb"], [0, 0]),
            ):
                text = StringTensor.from_list(values, offset_dtype=dtype)
                scores = torch.tensor(scores)
                expected = gather(text, scores)
                actual = compiled(text, scores)
                assert actual.tolist() == expected.tolist()
                assert actual._offset.dtype == expected._offset.dtype
                assert actual.size() == expected.size()
                torch.testing.assert_close(actual._data, expected._data)
            print(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "dtype": str(dtype),
                        "fullgraph": fullgraph,
                        "status": "pass",
                    }
                ),
                flush=True,
            )
        except Exception as e:
            print(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "dtype": str(dtype),
                        "fullgraph": fullgraph,
                        "status": "fail",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
