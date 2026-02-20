#!/bin/bash

CONFIG_DIR="./configs/lora"
SCRIPT="python eval_vlm.py"

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
    rm -rf ./artifacts
done