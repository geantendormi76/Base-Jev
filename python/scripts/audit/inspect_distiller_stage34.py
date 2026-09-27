# -*- coding: utf-8 -*-
from pathlib import Path

PIPELINE_PATH = Path("C:/dev/distiller/scripts/pipeline/run_distillation_pipeline.py")

def main():
    if not PIPELINE_PATH.is_file():
        print(f"[X] 文件不存在: {PIPELINE_PATH}")
        return
    lines = PIPELINE_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    print("=" * 80)
    print(" 🔬 正在探查 run_distillation_pipeline.py 阶段 3 与阶段 4 (110~185 行):")
    print("=" * 80)
    for idx in range(109, min(len(lines), 185)):
        print(f"{idx+1:3d} | {lines[idx]}")
    print("=" * 80)

if __name__ == "__main__":
    main()
