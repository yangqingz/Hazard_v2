#!/bin/bash
# filepath: /home/yangqingzheng/HAZARD/src/HAZARD/data/run.sh

assets_dir="assets"
mkdir -p "$assets_dir"

for i in {0..26}; do
    logfile="/home/yangqingzheng/HAZARD/src/HAZARD/data/room_setup_wind/suburb_scene_2023-4-$i/log.txt"
    if [[ -f "$logfile" ]]; then
        grep -oP '"url":\s*"\K[^"]+' "$logfile" | sed 's/private/public/g' | while read -r url; do
            wget -nc -P "$assets_dir" "$url"
        done
    fi
done