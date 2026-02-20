#!/bin/bash

CONFIG_DIR="./configs/sweep_params"

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running FT for $config"
    version=$(yq -r '.version' "$config")
    if [[ "$version" == unsloth ]]; then
        SCRIPT="train_unsloth.py"
    elif [[ "$version" == peft ]]; then
        SCRIPT="train_peft.py"
    else
        echo "Unknown version '$version' in $config — skipping"
        continue
    fi
    python $SCRIPT --config "$config"
done
