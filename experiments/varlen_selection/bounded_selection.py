import json, traceback, torch
from sdm import StringTensor
from sdm.tensor.var_len import _compact


def select(categories, mask):
    selected = mask.nonzero().flatten()
    physical = selected * categories.stride(0) + categories._storage_offset
    offset, index = _compact(
        categories._offset[physical],
        categories._offset[physical + 1],
        max_total=categories.numel() * categories._data.numel(),
    )
    return StringTensor(
        data=categories._data[index],
        offset=offset,
        valid=categories._valid[physical]
        if categories._valid is not None
        else None,
        size=selected.shape,
    )


for fullgraph in (False, True):
    for stride in (0, 1, 2):
        torch._dynamo.reset()
        compiled = torch.compile(select, fullgraph=fullgraph, dynamic=True)
        try:
            for values, mask in (
                (
                    ["unused", "a", None, "long", "red", "", "z", "q", "u"],
                    [1, 0, 1, 1],
                ),
                (
                    ["other", "longer", "qq", "b", "red", "", "z", "r", "u"],
                    [0, 1, 0, 1],
                ),
                (
                    ["other", "a", None, "long", "red", "", "z", "q", "u"],
                    [0, 0, 0, 0],
                ),
            ):
                source = StringTensor.from_list(
                    values, offset_dtype=torch.int32
                )
                cat = StringTensor(
                    data=source._data,
                    offset=source._offset,
                    valid=source._valid,
                    size=(4,),
                    stride=(stride,),
                    storage_offset=1,
                )
                mask = torch.tensor(mask)
                expected = cat.index_select(0, mask.nonzero().flatten())
                actual = compiled(cat, mask)
                assert actual.tolist() == expected.tolist()
                assert actual._offset.dtype == expected._offset.dtype
                torch.testing.assert_close(actual._data, expected._data)
            print(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "fullgraph": fullgraph,
                        "stride": stride,
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
                        "fullgraph": fullgraph,
                        "stride": stride,
                        "status": "fail",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
