#!/usr/bin/env bash
# 兼容 Zsh 与 Bash 的脚本路径动态解析
if [ -n "${ZSH_VERSION:-}" ]; then
    CURRENT_SCRIPT="${(%):-%x}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    CURRENT_SCRIPT="${BASH_SOURCE[0]}"
else
    CURRENT_SCRIPT="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "${CURRENT_SCRIPT}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SITE_DIR="${PROJECT_ROOT}/python/.venv/lib/python3.12/site-packages"

if [ -d "${SITE_DIR}" ]; then
    NVIDIA_LIB_PATHS="$(find "${SITE_DIR}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd ':' -)"
    ORT_CAPI_DIR="${SITE_DIR}/onnxruntime/capi"
    ACTUAL_ORT_SO="$(find "${ORT_CAPI_DIR}" -name "libonnxruntime.so*" -type f 2>/dev/null | head -n 1)"
    
    export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${ORT_CAPI_DIR}:${LD_LIBRARY_PATH:-}"
    if [ -n "${ACTUAL_ORT_SO}" ]; then
        export ORT_DYLIB_PATH="${ACTUAL_ORT_SO}"
        ln -sf "${ACTUAL_ORT_SO}" "${ORT_CAPI_DIR}/libonnxruntime.so" 2>/dev/null
    fi
fi
