# -*- coding: utf-8 -*-
"""
inspect_curator_filter.py
只读探查：查看 PRismCurator 的过滤契约与是否支持端点排除
"""
from __future__ import annotations

from pathlib import Path

CURATOR_PATH = Path("C:/dev/distiller/src-tauri/python/src/curator.py")

def main():
    print("=" * 80)
    print(" 🔬 正在探查 PRismCurator.curate_report() 过滤逻辑与排除能力...")
    print("=" * 80)

    if not CURATOR_PATH.is_file():
        print(f"[❌] 文件不存在: {CURATOR_PATH}")
        return

    lines = CURATOR_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_curate_report = False
    curate_def_lines = []

    for idx, line in enumerate(lines, 1):
        if "def curate_report" in line:
            in_curate_report = True
        if in_curate_report:
            curate_def_lines.append(f"{idx:3d} | {line}")
            # 捕获函数体前 45 行
            if len(curate_def_lines) >= 45:
                break

    print("\n".join(curate_def_lines))
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
