#!/bin/bash

Q3_DIR="./configs/zero_shot/qwen_3B"
SCRIPT="python eval_unsloth.py"

for config in "$Q3_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
done