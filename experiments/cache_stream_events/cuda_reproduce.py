"""Check CUDA cache transfer and scalar assignment with both wait forms."""

import json

import torch


def _make_function(transfer_stream, use_events):
    def compute(cache, mask):
        compute_stream = torch.cuda.current_stream()
        with torch.cuda.stream(transfer_stream):
            cache = cache.to("cuda", non_blocking=True)
        if use_events:
            compute_stream.wait_event(transfer_stream.record_event())
        else:
            compute_stream.wait_stream(transfer_stream)
        result = cache.square()
        result[mask] = 0.0
        cache.record_stream(compute_stream)
        return result

    return compute


for offloaded in (False, True):
    for use_events in (False, True):
        for fullgraph in (False, True):
            torch._dynamo.reset()
            function = _make_function(torch.cuda.Stream(), use_events)
            compiled = torch.compile(
                function, backend="inductor", fullgraph=fullgraph, dynamic=True
            )
            result = {
                "torch": torch.__version__,
                "offloaded": offloaded,
                "events": use_events,
                "fullgraph": fullgraph,
            }
            try:
                for rows in (32, 128, 7):
                    cache = torch.arange(rows, dtype=torch.float32)
                    cache = cache.pin_memory() if offloaded else cache.cuda()
                    mask = torch.arange(rows, device="cuda").remainder(3) == 0
                    expected = function(cache, mask)
                    actual = compiled(cache, mask)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        actual, expected, atol=0, rtol=0
                    )
            except Exception as error:  # noqa: BLE001
                result.update(status="fail", error=str(error))
            else:
                result.update(status="pass")
            print(json.dumps(result), flush=True)  # noqa: T201
