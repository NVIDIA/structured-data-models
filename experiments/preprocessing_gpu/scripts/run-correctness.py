"""Measure eager and compiled prediction on exactly the same fitted state."""
# ruff: noqa: D103, T201, BLE001

import argparse
from collections import Counter
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models import KumoRelational, KumoTabular


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["tabular", "relational"], required=True
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
    )
    parser.add_argument("--entry", choices=["inner", "predict"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--autocast", choices=["off", "bf16"], default="off")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--single-stream", action="store_true",
                        help="Diagnostic: use the compute stream for cache transfers")
    parser.add_argument("--capture-dynamic-outputs", action="store_true")
    parser.add_argument("--recompile-limit", type=int)
    parser.add_argument("--estimators", type=int, default=1)
    parser.add_argument("--context-rows", type=int, default=32)
    parser.add_argument(
        "--query-rows", type=int, nargs="+", default=[32, 128, 32]
    )
    parser.add_argument("--arm-index", type=int, default=0)
    parser.add_argument(
        "--query-indices", type=int, nargs="+", default=[1, 0, 2, 1]
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    result = {
        "config": vars(args),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "source": None,
        "weight_dtype": "float32",
        "autocast_dtype": "bfloat16" if args.autocast == "bf16" else None,
        "samples": [],
    }
    predictions = {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_text(json.dumps(result, indent=2) + "\n")

    def measure(function, query, related):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
            }
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        actual = function(query, related).numerical
        if cuda:
            end_event.record()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        metrics = {
            "wall_ms": elapsed * 1000,
            "graphs": dict(torch._dynamo.utils.counters["stats"]),
        }
        if cuda:
            peak = {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
            metrics.update(
                cuda_ms=start_event.elapsed_time(end_event),
                memory_before_bytes=before,
                memory_peak_bytes=peak,
                peak_extra_allocated_bytes=peak["allocated"]
                - before["allocated"],
            )
        # Copy after timing/peak collection; reference tensors stay on CPU.
        return actual.detach().cpu().clone(), metrics

    try:
        source_root = Path(__file__).resolve().parents[2]
        receipt = source_root / "SOURCE_COMMIT"
        if args.source_commit:
            result["source"] = args.source_commit
        elif receipt.is_file():
            result["source"] = receipt.read_text().strip()
        else:
            result["source"] = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        torch.set_num_threads(1)
        torch.manual_seed(123)
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            result["gpu"] = torch.cuda.get_device_name()
            result["gpu_capability"] = list(torch.cuda.get_device_capability())
            result["gpu_total_memory"] = torch.cuda.get_device_properties(
                0
            ).total_memory
            result["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
            result["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
            if args.autocast == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 requested but unsupported")
        if args.model == "tabular":
            data = np.load(args.data)
            context_ids = data["train_ids"][: args.context_rows]
            cx = TableTensor.from_tensor(
                torch.tensor(data["x"][context_ids])
            ).to(device)
            target = torch.tensor(data["y"][context_ids]).view(-1, 1)
            cy = TableTensor.from_tensor(
                CategoricalTensor.from_tensor(target.long())
                if args.task == "classification"
                else target.float()
            ).to(device)
            cr = None
            queries = [
                (
                    rows,
                    TableTensor.from_tensor(
                        torch.tensor(data["x"][data["validation_ids"][:rows]])
                    ).to(device),
                    None,
                )
                for rows in args.query_rows
            ]
            model = KumoTabular(
                task=args.task, size="small", pretrained=False, device=device
            )
            fit_options = {}
        else:
            bundle = torch.load(
                args.data, weights_only=False, map_location="cpu"
            )
            arm = bundle["arms"][args.arm_index]
            cx, cy, cr = [
                arm[key].to(device)
                for key in ["context", "context_target", "related_context"]
            ]
            queries = [
                (
                    index,
                    arm["queries"][index]["x"].to(device),
                    arm["queries"][index]["related"].to(device),
                )
                for index in args.query_indices
            ]
            model = KumoRelational(
                task=args.task, pretrained=False, device=device
            )
            fit_options = {"num_hops": arm["num_hops"]}
        result["actual_query_rows"] = [
            query.size(-2) for _, query, _ in queries
        ]
        model.models[args.task].load_state_dict(
            torch.load(
                args.checkpoint, weights_only=True, map_location=device
            ),
            strict=True,
        )
        model.eval()
        options = {}
        if args.capture_dynamic_outputs:
            options["capture_dynamic_output_shape_ops"] = True
        if args.recompile_limit is not None:
            options["cache_size_limit"] = args.recompile_limit
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.autocast == "bf16",
            ),
            torch._dynamo.config.patch(options),
        ):
            model.fit(
                cx,
                cy,
                cr,
                generator=torch.Generator(device=device).manual_seed(123),
                num_estimators=args.estimators,
                **fit_options,
            )
            result["eager_fit_succeeded"] = True
            result["cache_num_batches"] = model._cache["num_batches"]
            plain_cpu = [t for t in model._cache._tensors()
                         if type(t) is torch.Tensor and t.device.type == "cpu"]
            result["plain_cpu_cache_tensors"] = len(plain_cpu)
            result["pinned_plain_cpu_cache_tensors"] = sum(t.is_pinned() for t in plain_cpu)
            result["cache_devices"] = dict(Counter(
                str(t.device) for t in model._cache._tensors()))
            if args.single_stream and cuda:
                stream = torch.cuda.current_stream()
                model._transfer_streams[torch.device("cuda:0")] = stream
            references = {}
            for phase in ["eager", "compiled"]:
                if phase == "compiled":
                    torch._dynamo.utils.counters.clear()
                    compile_options = {
                        "backend": args.backend,
                        "fullgraph": args.fullgraph,
                        "dynamic": True,
                    }
                    if args.entry == "inner":
                        for inner in model.models.values():
                            inner.compile(**compile_options)
                        predict = model.predict
                    else:
                        predict = torch.compile(
                            model.predict, **compile_options
                        )
                else:
                    predict = model.predict
                for case, query, related in queries:
                    for repetition in range(-1 - args.warmups, args.repeats):
                        actual, metrics = measure(predict, query, related)
                        if phase == "eager":
                            references[case] = actual
                        expected = references[case]
                        close = torch.isclose(
                            actual, expected, atol=1e-5, rtol=1e-4
                        )
                        sample = {
                            "phase": phase,
                            "case": case,
                            "rows": query.size(-2),
                            "requested_rows": case
                            if args.model == "tabular"
                            else None,
                            "repetition": repetition,
                            "kind": "first"
                            if repetition == -1 - args.warmups
                            else ("warmup" if repetition < 0 else "warm"),
                            "output_dtype": str(actual.dtype),
                            "max_abs_error": (actual - expected)
                            .abs()
                            .max()
                            .item(),
                            "parity": bool(close.all()),
                            "failed_values": int((~close).sum()),
                            "total_values": actual.numel(),
                            **metrics,
                        }
                        if args.task == "classification":
                            sample["class_agreement"] = float(
                                (actual.argmax(-1) == expected.argmax(-1))
                                .float()
                                .mean()
                            )
                        if args.save_predictions:
                            predictions[f"{phase}-{case}-{len(predictions)}"] = {
                                "actual": actual, "reference": expected,
                                "case": case, "phase": phase,
                                "repetition": repetition}
                            torch.save(predictions, output.with_suffix(".predictions.pt"))
                        result["samples"].append(sample)
                        save()
            result["graph_breaks"] = dict(
                torch._dynamo.utils.counters["graph_break"]
            )
            result["status"] = (
                "pass"
                if all(v["parity"] for v in result["samples"])
                else "parity_fail"
            )
    except Exception as error:
        result.update(
            status="fail",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    save()
    print(
        json.dumps({"status": result["status"], "output": str(output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
