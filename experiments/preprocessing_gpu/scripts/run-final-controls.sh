#!/bin/bash
set -uo pipefail
root=/validation
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
run_case() {
 name=$1; runtime=$2; source_dir=$3; runner=$4; shift 4
 cd "$root/$source_dir"
 export TORCHINDUCTOR_CACHE_DIR="$root/cache/$name"
 PYTHONPATH=. timeout --signal=TERM --kill-after=20s 300 "$root/$runtime/bin/python" "$root/$runner" "$@" --output "$root/results/$name.json" > "$root/results/$name.log" 2>&1
 printf 'process_exit=%s\n' "$?" >> "$root/results/$name.log"
}
args=(--device cuda --model tabular --task classification --context-rows 256 --data "$root/data/classification.npz" --checkpoint "$root/data/tabular-classifier.pt" --save-predictions)
run_case tabular27-context256-predict-fp32-partial-events18 torch27 repo-events run-correctness.py "${args[@]}" --source-commit c75ba3af6+18a4d151a --entry predict --autocast off --estimators 1 --query-rows 128
run_case tabular-bf16-inner-backend-eager torch214 repo-emptyfix run-correctness.py "${args[@]}" --source-commit c75ba3af6+18a4d151a+4071789e4 --entry inner --autocast bf16 --estimators 1 --query-rows 32 --backend eager --fullgraph
run_case tabular-bf16-inner-emulate-casts torch214 repo-emptyfix gpu_emulate_casts.py --runner "$root/run-correctness.py" "${args[@]}" --source-commit c75ba3af6+18a4d151a+4071789e4 --entry inner --autocast bf16 --estimators 1 --query-rows 32 --fullgraph
run_case offload2-inner-control torch214 repo-events run-correctness.py "${args[@]}" --source-commit c75ba3af6+18a4d151a --entry inner --autocast off --estimators 2 --query-rows 128 --fullgraph
touch "$root/results/final-controls.complete"
