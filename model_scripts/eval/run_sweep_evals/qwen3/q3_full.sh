#!/bin/bash

Q3_DIR="./configs/sweep_evals/qwen3/qwen3_full"
SCRIPT="python eval_vlm.py"

for config in "$Q3_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
done