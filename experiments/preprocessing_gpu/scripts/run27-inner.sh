#!/bin/bash
set -uo pipefail
root=/validation
while ! test -f "$root/results/source-control.complete"; do sleep 2; done
cd "$root/repo-events"
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCHINDUCTOR_CACHE_DIR="$root/cache/tabular27-inner-full"
PYTHONPATH=. timeout --signal=TERM --kill-after=15s 240 "$root/torch27/bin/python" "$root/run-correctness.py" --device cuda --source-commit c75ba3af6+18a4d151a --save-predictions --model tabular --task classification --entry inner --autocast off --fullgraph --estimators 1 --context-rows 256 --query-rows 128 --data "$root/data/classification.npz" --checkpoint "$root/data/tabular-classifier.pt" --output "$root/results/tabular27-inner-fp32-full.json" > "$root/results/tabular27-inner-fp32-full.log" 2>&1
printf 'process_exit=%s\n' "$?" >> "$root/results/tabular27-inner-fp32-full.log"
touch "$root/results/inner27.complete"
