#!/bin/bash

Q7_DIR="./configs/zero_shot/qwen_7B"
SCRIPT="python eval_vlm.py"

for config in "$Q7_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
done