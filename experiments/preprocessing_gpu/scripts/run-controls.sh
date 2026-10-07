#!/bin/bash
set -uo pipefail
root=/validation
results=$root/results
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
run_case() {
    name=$1; source_dir=$2; source=$3
    shift 3
    cd "$root/$source_dir"
    export TORCHINDUCTOR_CACHE_DIR="$root/cache/$name"
    date -u +%FT%TZ >> "$results/$name.log"
    PYTHONPATH=. timeout --signal=TERM --kill-after=30s 480 "$root/torch214/bin/python" "$root/run-correctness.py" --device cuda --source-commit "$source" --save-predictions "$@" --output "$results/$name.json" >> "$results/$name.log" 2>&1
    printf 'process_exit=%s\n' "$?" >> "$results/$name.log"
    date -u +%FT%TZ >> "$results/$name.log"
}
run_case offload2-single-stream-control repo-events c75ba3af6+18a4d151a --model tabular --task classification --entry predict --single-stream --autocast off --fullgraph --estimators 2 --context-rows 256 --query-rows 128 --data "$root/data/classification.npz" --checkpoint "$root/data/tabular-classifier.pt"
run_case relational-final-e2abde-fp32-full repo-final e2abde1542f15ce605b33739411a52edbc87d564 --model relational --entry predict --autocast off --fullgraph --capture-dynamic-outputs --arm-index 0 --query-indices 2 --data "$root/data/driver-dnf_bundle.pt" --checkpoint "$root/data/relational-classifier.pt"
run_case tabular-bf16-inner-control repo-emptyfix c75ba3af6+18a4d151a+4071789e4 --model tabular --task classification --entry inner --autocast bf16 --fullgraph --estimators 1 --context-rows 256 --query-rows 32 128 32 --data "$root/data/classification.npz" --checkpoint "$root/data/tabular-classifier.pt"
touch "$results/controls.complete"
