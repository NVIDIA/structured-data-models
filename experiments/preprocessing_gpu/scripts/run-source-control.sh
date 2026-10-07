#!/bin/bash
set -uo pipefail
root=/validation
while ! test -f "$root/results/final-controls.complete"; do sleep 3; done
cp -a "$root/repo-events" "$root/repo-currentstream"
cd "$root/repo-currentstream"
patch -p1 < "$root/compiled-current-stream-f0a5db1e4.patch"
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
export TORCHINDUCTOR_CACHE_DIR="$root/cache/offload2-single-stream-control"
PYTHONPATH=. timeout --signal=TERM --kill-after=20s 300 "$root/torch214/bin/python" "$root/run-correctness.py" --device cuda --source-commit c75ba3af6+18a4d151a+f0a5db1e4 --save-predictions --model tabular --task classification --entry predict --autocast off --fullgraph --estimators 2 --context-rows 256 --query-rows 128 --data "$root/data/classification.npz" --checkpoint "$root/data/tabular-classifier.pt" --output "$root/results/offload2-currentstream-source.json" > "$root/results/offload2-currentstream-source.log" 2>&1
printf 'process_exit=%s\n' "$?" >> "$root/results/offload2-currentstream-source.log"
touch "$root/results/source-control.complete"
