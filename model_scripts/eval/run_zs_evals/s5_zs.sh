#!/bin/bash

S5_DIR="./configs/zero_shot/smolvlm"
SCRIPT="python eval_peft.py"

for config in "$S5_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
done