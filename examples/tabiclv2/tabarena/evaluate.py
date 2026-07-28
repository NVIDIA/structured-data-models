r"""Evaluate TabICLv2 results on TabArena.

$ uv run --group example-tabarena python examples/tabiclv2/tabarena/evaluate.py
"""

from pathlib import Path

from tabarena.contexts import TabArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

example_dir = Path(__file__).parent.parent
result_dir = example_dir / "tabarena_out" / "TabICLv2"
output_dir = example_dir / "evals" / "TabICLv2"
output_dir.mkdir(parents=True, exist_ok=True)

base_context = TabArenaContext()
method_metadata = MethodMetadata.baseline(
    method="SDMTabICLv2System_c1_default",
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
results.to_csv(output_dir / "method_results_per_split.csv", index=False)

methods = processed.to_method_metadata_lst(
    new_result_prefix="[SDM] ",
)
context = TabArenaContext.from_new_methods(methods)
leaderboard = context.compare(output_dir=output_dir)
website = context.leaderboard_to_website_format(leaderboard)
print(website.to_string(index=False))
