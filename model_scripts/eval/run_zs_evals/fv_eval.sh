#!/bin/bash

FV_DIR="./configs/zero_shot/frozen_vision"
SCRIPT="python eval_vlm.py"
PEFT_SCRIPT="python eval_peft.py"

for config in "$FV_DIR"/*.yaml; do
    echo "Running evaluation for $config"
    $SCRIPT --config "$config"
    rm -rf ./artifacts
done