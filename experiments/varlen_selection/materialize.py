import json, torch, traceback
from sdm import StringTensor


def clone(text):
    return text.clone()


def clone_slice(text):
    return text[1::2].clone()


def cat(text):
    return torch.cat([text, text])


for fn in (clone, clone_slice, cat):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=fullgraph, dynamic=True)
        try:
            for values in (
                ["a", None, "longer"],
                ["xx", "", None, "y", "green"],
                [],
                ["", "a"],
            ):
                text = StringTensor.from_list(values)
                expected = fn(text)
                actual = compiled(text)
                assert actual.tolist() == expected.tolist()
                assert actual.stride() == expected.stride()
                assert actual._offset.dtype == expected._offset.dtype
            print(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "fn": fn.__name__,
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
                        "fn": fn.__name__,
                        "fullgraph": fullgraph,
                        "status": "fail",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
