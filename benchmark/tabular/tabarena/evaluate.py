r"""Evaluate SDM tabular model results on TabArena."""

import argparse
from pathlib import Path

from tabarena.contexts import TabArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

from benchmark.tabular.model import MODEL_CONFIGS

benchmark_dir = Path(__file__).parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output_root",
    type=Path,
    default=benchmark_dir / "tabarena_out",
    help="Directory holding one result directory per run.",
)
parser.add_argument(
    "--name",
    action="append",
    help="Named run to score (repeatable; default: the built-in models).",
)
parser.add_argument("--subset", help="TabArena task subset.")
args = parser.parse_args()
result_root = args.output_root
output_root = result_root.parent / "evals"

# (label, method, result_dir)
runs = [
    (name, "SDMKumoTabular_c1_default", result_root / name / "outer_model")
    for name in args.name or []
] or [
    (
        model_config.name,
        model_config.tabarena_method_name,
        result_root / model_config.name / "outer_model",
    )
    for model_config in MODEL_CONFIGS.values()
]
runs = [
    run for run in runs if next(run[2].rglob("results.pkl"), None) is not None
]
if not runs:
    raise FileNotFoundError(f"No TabArena results found under {result_root}")

base_context = TabArenaContext()
methods = []
for label, method, result_dir in runs:
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    method_metadata = MethodMetadata.baseline(
        method=method,
        compute="gpu",
        artifact_dir=output_dir / "artifacts",
    )
    processed = EndToEnd.from_path_raw(
        path_raw=result_dir,
        method_metadata=method_metadata,
        task_metadata=base_context.task_metadata_collection,
        backend="native",
    )
    prefix = f"[{label}] "
    results = processed.get_results(new_result_prefix=prefix)
    results.to_csv(
        output_dir / "method_results_per_split.csv",
        index=False,
    )
    methods.extend(processed.to_method_metadata_lst(new_result_prefix=prefix))

context = TabArenaContext.from_new_methods(methods)
leaderboard = context.compare(
    output_dir=output_root,
    subset=args.subset,
    only_valid_tasks=[method.method for method in methods],
)
website = context.leaderboard_to_website_format(leaderboard)
print(website.to_string(index=False))
