#!/usr/bin/env sh
set -eu

for seed in 7 8 9; do
  python files/main.py \
    --epochs 10 \
    --train-samples 4000 \
    --eval-samples 1000 \
    --image-size 64 \
    --max-radius 20 \
    --seed "$seed" \
    --model-path "artifacts/model-seed-${seed}.keras" \
    --metrics-path "artifacts/metrics-seed-${seed}.json" \
    --plot-path "artifacts/predictions-seed-${seed}.png"
done
