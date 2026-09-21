# -*- coding: utf-8 -*-
from pathlib import Path

CONTRACTS_PATH = Path("C:/dev/distiller/src-tauri/python/src/contracts.py")

def main():
    if not CONTRACTS_PATH.is_file():
        print(f"[X] 文件不存在: {CONTRACTS_PATH}")
        return
    lines = CONTRACTS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    print("=" * 80)
    print(f" 🔬 正在探查 distiller/src/contracts.py 中 Curator 契约定义:")
    print("=" * 80)
    capturing = False
    captured = []
    for idx, line in enumerate(lines, 1):
        if "class CuratorStats" in line or "class CuratedEndpoint" in line:
            capturing = True
        if capturing:
            captured.append(f"{idx:3d} | {line}")
            if len(captured) >= 40 and line.strip().startswith("class "):
                break
            if len(captured) >= 60:
                break
    print("\n".join(captured))
    print("=" * 80)

if __name__ == "__main__":
    main()
