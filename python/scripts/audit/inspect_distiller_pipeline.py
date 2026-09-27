# -*- coding: utf-8 -*-
"""
inspect_distiller_pipeline.py
只读探查：深度剖析 C:/dev/distiller 核心数据生产链路
职责：解析数据流水线脚本的输入源、生成参数与为什么输出只有 2 行
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

DISTILLER_ROOT = Path("C:/dev/distiller")

TARGET_SCRIPTS = [
    DISTILLER_ROOT / "scripts" / "pipeline" / "generate_dual_orthogonal_dataset.py",
    DISTILLER_ROOT / "scripts" / "pipeline" / "run_distillation_pipeline.py",
    DISTILLER_ROOT / "src-tauri" / "python" / "src" / "curator.py",
    DISTILLER_ROOT / "src-tauri" / "python" / "src" / "synthesizer.py",
]

def analyze_script(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        return {"exists": False}
    
    content = p.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    # 提取 argparse 参数配置
    arg_lines = [line.strip() for line in lines if "add_argument" in line]
    # 提取默认输入输出路径
    defaults = re.findall(r'default\s*=\s*["\']([^"\']+)["\']', content)
    # 提取路径特征与常量
    paths_in_code = sorted(list(set(re.findall(r'["\'](?:[^"\']+[/\\][^"\']+\.(?:jsonl|json|py))["\']', content))))

    # 寻找可能的数量限制 (limit, slice, [:N])
    limits = re.findall(r'(\w*limit\w*\s*=\s*\d+|\[:\s*\d+\s*\]|max_samples\s*=\s*\d+)', content)

    return {
        "exists": True,
        "total_lines": len(lines),
        "argparse_args": arg_lines[:8],
        "default_paths": defaults,
        "explicit_paths_in_code": paths_in_code,
        "limit_patterns": limits[:8],
        "code_head": [line for line in lines[:30] if line.strip() and not line.strip().startswith("#")][:10],
    }

def main():
    print("=" * 85)
    print(" 🔬 正在深度解构 C:/dev/distiller 上游数据生产管线...")
    print("=" * 85)

    for p in TARGET_SCRIPTS:
        info = analyze_script(p)
        print(f"\n[📜 脚本解析]: {p.name} ({p.relative_to(DISTILLER_ROOT)})")
        if not info["exists"]:
            print("  └─ [❌] 文件不存在！")
            continue
        
        print(f"  ├─ 总行数           : {info['total_lines']}")
        print(f"  ├─ 明确引用的路径   : {info['explicit_paths_in_code']}")
        print(f"  ├─ 默认参数路径     : {info['default_paths']}")
        if info["limit_patterns"]:
            print(f"  ├─ 数量限制特征匹配 : {info['limit_patterns']}")
        if info["argparse_args"]:
            print(f"  ├─ CLI 参数定义     :")
            for arg in info["argparse_args"]:
                print(f"     • {arg}")
        print(f"  └─ 核心逻辑前瞻     :")
        for ch in info["code_head"][:5]:
            print(f"     | {ch}")

    print("\n" + "=" * 85)
    print(" [✓] 探查完毕，零写入，零数据修改。")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
