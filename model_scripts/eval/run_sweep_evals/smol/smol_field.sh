#!/bin/bash

Q3_DIR="./configs/sweep_evals/smol/smol_field"
SCRIPT="python eval_peft.py"

for config in "$Q3_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
done