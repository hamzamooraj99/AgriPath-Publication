#!/bin/bash

CONFIG_DIR="./configs/alpha_ratio_1"
SCRIPT="python alpha_mod.py"

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
done