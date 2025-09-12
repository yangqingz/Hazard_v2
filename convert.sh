#!/bin/bash

BASE_DIR="/home/yangqingzheng/HAZARD/outputs/wind/llm_gpt5_train"
OUTPUT_FILE="train2.jsonl"
START_NUM=10
END_NUM=26

for i in $(seq $START_NUM $END_NUM); do
    input_dir="$BASE_DIR/suburb_scene_2023-0-$i/0"
    echo "Processing: $input_dir"
    python3 convert_data.py "$input_dir" "$OUTPUT_FILE"
done

echo "Conversion complete!"