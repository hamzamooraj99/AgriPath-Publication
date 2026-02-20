#!/bin/bash

CONFIG_DIR="./configs/lab_eval"
SCRIPT="python field_eval.py"

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
    rm -rf ./artifacts
done