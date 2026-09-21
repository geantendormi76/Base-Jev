# -*- coding: utf-8 -*-
"""
inspect_pipeline_sources.py
深度只读探查：寻找真实主力训练集的生成源头与上游资产
职责：扫描 base-jev/scripts/pipeline/、全工程数据引用及 C:/dev/distiller 资产
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_JEV_ROOT = Path("C:/dev/base-jev")
INVAR_ROOT = Path("C:/dev/Invar")
DISTILLER_ROOT = Path("C:/dev/distiller")
PIPELINE_DIR = BASE_JEV_ROOT / "scripts" / "pipeline"

def get_file_summary(path: Path) -> Dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {"error": str(exc)}
    
    lines = content.splitlines()
    docstring = ""
    for line in lines[:15]:
        clean = line.strip()
        if clean.startswith('"""') or clean.startswith("'''") or clean.startswith("#"):
            docstring += clean + " "
    
    # 提取代码中引用的数据集路径
    referenced_datasets = set(re.findall(r"[\w\-_/\\\.]+\.jsonl", content))
    has_dual_projection = "generate_dual" in content or "target_scores" in content
    has_distilled_ref = "distilled_invar" in content

    return {
        "file_name": path.name,
        "size_kb": round(path.stat().st_size / 1024, 2),
        "total_lines": len(lines),
        "docstring_preview": docstring[:120].strip() if docstring else "无注释说明",
        "referenced_jsonl": sorted(list(referenced_datasets)),
        "has_dual_logic": has_dual_projection,
        "has_distilled_ref": has_distilled_ref,
    }

def main():
    print("=" * 85)
    print(" 🔍 base-jev: 主力训练集生成源头与流水线全景只读探查")
    print(f" 📂 base-jev 流水线目录 : {PIPELINE_DIR.resolve()}")
    print(f" 📂 上游蒸馏车间目录   : {DISTILLER_ROOT.resolve()}")
    print("=" * 85)

    # 1. 扫描 base-jev/scripts/pipeline/ 目录下所有脚本
    print("\n[1] 正在剖析 base-jev/scripts/pipeline/ 下的所有流水线脚本...")
    if not PIPELINE_DIR.is_dir():
        print(f"[❌] 目录不存在: {PIPELINE_DIR}")
    else:
        pipeline_files = sorted(list(PIPELINE_DIR.glob("*.py")))
        print(f"  ├─ 发现脚本总数: {len(pipeline_files)} 个")
        for f in pipeline_files:
            info = get_file_summary(f)
            print(f"\n  • 脚本文件: [{f.name}] ({info.get('size_kb')} KB, {info.get('total_lines')} 行)")
            print(f"    ├─ 功能摘要    : {info.get('docstring_preview')}")
            print(f"    ├─ 关联 JSONL : {info.get('referenced_jsonl')}")
            print(f"    └─ 特征标记    : 双正交逻辑={info.get('has_dual_logic')}, 蒸馏引用={info.get('has_distilled_ref')}")

    # 2. 全工程搜索 HANDOFF.md 提及的 generate_dual_orthogonal_dataset.py
    print("\n[2] 正在全工程物理检索 generate_dual_orthogonal_dataset.py 及相关生成脚本...")
    matched_scripts = list(BASE_JEV_ROOT.rglob("*dual_orthogonal*.py"))
    matched_scripts += list(BASE_JEV_ROOT.rglob("*generate*.py"))
    matched_scripts = sorted(list(set(matched_scripts)))
    
    if matched_scripts:
        print(f"  ├─ 命中候选生成脚本 ({len(matched_scripts)} 个):")
        for m in matched_scripts:
            rel = m.relative_to(BASE_JEV_ROOT)
            print(f"     • {rel} ({round(m.stat().st_size/1024, 2)} KB)")
    else:
        print("  └─ [!] base-jev 树下未发现任何包含 'dual_orthogonal' 的独立脚本！")

    # 3. 探查上游 C:/dev/distiller 目录是否存在及其资产
    print("\n[3] 正在探查上游数据策展车间 (C:/dev/distiller)...")
    if not DISTILLER_ROOT.is_dir():
        print(f"  └─ [!] 上游目录不存在: {DISTILLER_ROOT}")
    else:
        print(f"  ├─ [✓] 成功探测到上游目录: {DISTILLER_ROOT}")
        all_distiller_files = list(DISTILLER_ROOT.rglob("*"))
        distiller_py = [p for p in all_distiller_files if p.is_file() and p.suffix == ".py"]
        distiller_jsonl = [p for p in all_distiller_files if p.is_file() and p.suffix in [".jsonl", ".json"]]
        
        print(f"  ├─ Python 模块数 : {len(distiller_py)} 个")
        print(f"  ├─ 数据资产文件数 : {len(distiller_jsonl)} 个")
        
        if distiller_jsonl:
            print("\n  [📊 distiller 内部数据文件物理清单]:")
            for dj in distiller_jsonl:
                rel = dj.relative_to(DISTILLER_ROOT)
                lines = 0
                try:
                    with dj.open("r", encoding="utf-8", errors="ignore") as f:
                        lines = sum(1 for line in f if line.strip())
                except Exception:
                    pass
                print(f"     • {str(rel):40s} : {lines:6d} 行 ({round(dj.stat().st_size/1024, 2)} KB)")

        if distiller_py:
            print("\n  [📜 distiller 核心 Python 脚本]:")
            for dp in distiller_py[:10]:
                rel = dp.relative_to(DISTILLER_ROOT)
                print(f"     • {rel}")

    print("\n" + "=" * 85)
    print(" [✓] 探查完毕，零写入，零数据修改。")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
