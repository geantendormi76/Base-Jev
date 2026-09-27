# -*- coding: utf-8 -*-
"""
inspect_curator_tail.py
只读探查：查看 curate_report() 尾部实现及 CuratedEndpoint 契约
"""
from pathlib import Path

CURATOR_PATH = Path("C:/dev/distiller/src-tauri/python/src/curator.py")

def main():
    if not CURATOR_PATH.is_file():
        return
    lines = CURATOR_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    print("=" * 80)
    print(" 🔬 PRismCurator.curate_report() 395~440 行实现切片:")
    print("=" * 80)
    for idx in range(394, min(len(lines), 440)):
        print(f"{idx+1:3d} | {lines[idx]}")
    print("=" * 80)

if __name__ == "__main__":
    main()
