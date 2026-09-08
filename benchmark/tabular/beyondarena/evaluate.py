r"""Evaluate SDM tabular model results on BeyondArena."""

from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

benchmark_dir = Path(__file__).parent.parent
result_root = benchmark_dir / "beyondarena_out"
output_root = benchmark_dir / "beyondarena_evals"

runs = [
    result_dir
    for result_dir in sorted(result_root.iterdir())
    if result_dir.is_dir()
    and next(result_dir.rglob("results.pkl"), None) is not None
]

if not runs:
    raise FileNotFoundError(
        f"No BeyondArena results found under {result_root}"
    )

base_context = BeyondArenaContext()
for result_dir in runs:
    run_name = result_dir.name
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    method_metadata = MethodMetadata.baseline(
        method=(
            f"SDM{''.join(char for char in run_name if char.isalnum())}"
            "System_c1"
        ),
        compute="gpu",
        artifact_dir=output_dir / "artifacts",
    )
    processed = EndToEnd.from_path_raw(
        path_raw=result_dir,
        method_metadata=method_metadata,
        task_metadata=base_context.task_metadata_collection,
        backend="native",
    )
    results = processed.get_results(new_result_prefix="[SDM] ")
    results.to_csv(
        output_dir / "method_results_per_split.csv",
        index=False,
    )
    methods = processed.to_method_metadata_lst(new_result_prefix="[SDM] ")
    context = BeyondArenaContext.from_new_methods(methods)
    leaderboard = context.compare(
        output_dir=output_dir,
        only_valid_tasks=[method.method for method in methods],
    )
    website = context.leaderboard_to_website_format(leaderboard)
    website.to_csv(output_dir / "leaderboard.csv", index=False)
    print(f"\n{run_name}\n{website.to_string(index=False)}")
