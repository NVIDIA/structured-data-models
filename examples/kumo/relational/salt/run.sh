#!/usr/bin/env bash

TASKS=(
  plant
  shippingpoint
  itemincotermsclassification
  headerincotermsclassification
  salesoffice
  salesgroup
  customerpaymentterms
  shippingcondition
)

CONTEXT_SIZES=(
  10_000
  20_000
  30_000
  40_000
  50_000
  60_000
  70_000
  80_000
  90_000
  100_000
)

NUM_ESTIMATORS=(
  1
  8
)

for context_size in "${CONTEXT_SIZES[@]}"; do
  for num_estimators in "${NUM_ESTIMATORS[@]}"; do
    echo "${context_size} ${num_estimators}E"

    for task in "${TASKS[@]}"; do
      python main.py \
        --batch_size=100_000 \
        --context_size="${context_size}" \
        --num_estimators="${num_estimators}" \
        --task="${task}"
    done
  done
done
