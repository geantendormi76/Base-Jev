#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_DIR="${PROJECT_ROOT}/python/.venv/lib/python3.12/site-packages"

NVIDIA_LIB_PATHS="$(find "${SITE_DIR}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd ':' -)"
ORT_CAPI_DIR="${SITE_DIR}/onnxruntime/capi"
ACTUAL_ORT_SO="$(find "${ORT_CAPI_DIR}" -name "libonnxruntime.so*" -type f 2>/dev/null | head -n 1)"

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${ORT_CAPI_DIR}:${LD_LIBRARY_PATH:-}"
export ORT_DYLIB_PATH="${ACTUAL_ORT_SO}"
# 依据官方规范注入 LD_PRELOAD，保障 C++ 析构安全
export LD_PRELOAD="${ACTUAL_ORT_SO}:${LD_PRELOAD:-}"

echo "==========================================================================="
echo " 🚀 Base-Jev: 工业级出厂硬件质检启动器 (纯 Rust + CUDA TensorCore)"
echo " 📦 生效 ORT 物理库: ${ORT_DYLIB_PATH}"
echo " 📂 动态链接穿透项: $(echo "${NVIDIA_LIB_PATHS}" | tr ':' '\n' | wc -l) 个 NVIDIA 库目录"
echo "==========================================================================="

cd "${PROJECT_ROOT}"
cargo run --bin base-jev-qa -- "$@"
