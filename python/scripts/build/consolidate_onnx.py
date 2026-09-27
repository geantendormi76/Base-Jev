import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
ONNX_FILE = MODELS_DIR / "system_one_unified.onnx"
TARGET_DATA_NAME = "system_one_unified.onnx.data"

import onnx
from onnx.external_data_helper import load_external_data_for_model, convert_model_to_external_data

def consolidate_onnx():
    print("=" * 70)
    print(" 🛠️  正在执行 ONNX 外部权重单文件压铸归一化...")
    print(f" 📦 目标计算图: {ONNX_FILE}")
    print("=" * 70)

    # 1. 加载包含散落权重的全量模型到内存
    print("\n[1/3] 正在装配扫描 258 个离散权重切片...")
    model = onnx.load(str(ONNX_FILE), load_external_data=True)

    # 2. 重新保存为单一连续外部数据文件
    print(f"[2/3] 正在将其熔铸为标准单文件: {TARGET_DATA_NAME} (约 2.65 GB)...")
    temp_onnx = MODELS_DIR / "temp_consolidated.onnx"
    
    # 强制将所有张量合并到唯一的单一数据文件
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=TARGET_DATA_NAME,
        size_threshold=1024,  # 大于 1KB 的张量全部收拢入 data
    )
    
    onnx.save_model(model, str(temp_onnx))

    # 3. 物理替换并清理 258 个散乱的碎屑文件
    print("[3/3] 清理散落碎屑文件并锁定法定资产...")
    
    # 获取需要清理的散乱文件特征
    for p in MODELS_DIR.glob("*"):
        if p.is_file():
            if (p.name.startswith("backbone.") or p.name.startswith("onnx__MatMul_") or p.name.startswith("heads.")) and p.name != TARGET_DATA_NAME:
                try:
                    p.unlink()
                except Exception:
                    pass

    if ONNX_FILE.exists():
        ONNX_FILE.unlink()
    temp_onnx.rename(ONNX_FILE)

    data_file = MODELS_DIR / TARGET_DATA_NAME
    data_size_gb = data_file.stat().st_size / (1024 * 1024 * 1024)
    onnx_size_mb = ONNX_FILE.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print(" 🎉 ONNX 工业标准资产重组压铸圆满成功！")
    print(f" 📄 主计算图文件: {ONNX_FILE.name} ({onnx_size_mb:.2f} MB)")
    print(f" 💾 单一全量权重: {data_file.name} ({data_size_gb:.2f} GB)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    consolidate_onnx()
