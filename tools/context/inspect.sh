#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "用法: $0 <文件路径1> [文件路径2 ...]"
    exit 1
fi

echo "===== [CODE INSPECTION DUMP START] ====="
for target in "$@"; do
    if [ -f "$target" ]; then
        echo "--- FILE: $target ---"
        cat "$target"
        echo -e "\n"
    elif [ -d "$target" ]; then
        echo "--- DIRECTORY LISTING: $target ---"
        find "$target" -maxdepth 2 -not -path '*/.*'
        echo -e "\n"
    else
        echo "--- WARNING: 路径不存在: $target ---"
    fi
done
echo "===== [CODE INSPECTION DUMP END] ====="
