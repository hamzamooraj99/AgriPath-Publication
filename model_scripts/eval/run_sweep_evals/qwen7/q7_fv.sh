#!/bin/bash

Q3_DIR="./configs/sweep_evals/qwen7/qwen7_fv"
SCRIPT="python eval_vlm.py"

for config in "$Q3_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
done