# -*- coding: utf-8 -*-
"""
inspect_datasets.py
base-jev 数据集资产深度只读探查雷达
职责：全面清点 data/ 目录下全部数据集的物理开销、行数、数据契约原语及污染标记
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def compute_sha256_prefix(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()[:16]

def inspect_jsonl_file(path: Path) -> Dict[str, Any]:
    total_lines = 0
    valid_json_count = 0
    replay_tag_count = 0
    primitives_counter = Counter()
    schema_sample = {}
    has_target_scores = 0
    has_target_dist = 0
    
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for idx, raw_line in enumerate(f, 1):
            line_str = raw_line.strip()
            if not line_str:
                continue
            total_lines += 1
            if "GOLD-REPLAY" in line_str or "gold_replay" in line_str.lower():
                replay_tag_count += 1
            try:
                item = json.loads(line_str)
                valid_json_count += 1
                prim = item.get("primitive", "<None>")
                primitives_counter[prim] += 1
                if "target_scores" in item:
                    has_target_scores += 1
                if "target_distribution" in item:
                    has_target_dist += 1
                if not schema_sample and isinstance(item, dict):
                    schema_sample = {
                        "sample_id": item.get("sample_id"),
                        "primitive": prim,
                        "keys": sorted(list(item.keys())),
                        "has_impact_head_target": bool(item.get("target_scores", {}).get("impact")),
                        "has_sens_head_target": bool(item.get("target_scores", {}).get("sensitivity")),
                    }
            except Exception:
                pass

    return {
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "sha256_prefix": compute_sha256_prefix(path),
        "total_lines": total_lines,
        "valid_json_count": valid_json_count,
        "primitives_distribution": dict(primitives_counter),
        "replay_tag_count": replay_tag_count,
        "has_dual_target_scores_ratio": f"{has_target_scores}/{valid_json_count}",
        "has_single_dist_ratio": f"{has_target_dist}/{valid_json_count}",
        "schema_sample": schema_sample,
    }

def main():
    print("=" * 85)
    print(" 📂 base-jev: data/ 目录数据集资产全量只读探查雷达")
    print(f" 📁 目标目录: {DATA_DIR.resolve()}")
    print("=" * 85)

    if not DATA_DIR.is_dir():
        print(f"[❌] 目录不存在: {DATA_DIR}")
        sys.exit(1)

    all_files = sorted(list(DATA_DIR.glob("*")))
    if not all_files:
        print("[!] data/ 目录下未发现任何物理文件！")
        return

    jsonl_reports = []
    other_files = []

    for p in all_files:
        if not p.is_file():
            continue
        if p.suffix.lower() == ".jsonl":
            rep = inspect_jsonl_file(p)
            jsonl_reports.append(rep)
        else:
            other_files.append({
                "file_name": p.name,
                "suffix": p.suffix,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
                "sha256_prefix": compute_sha256_prefix(p),
            })

    # 1. 结构化表格渲染
    print("\n[📊 JSONL 核心数据集清单与物理事实]:")
    print(f"{'文件名':<40} {'行数':<8} {'大小(MB)':<10} {'原语分布 (Primitive)':<22} {'Replay污染':<10}")
    print("-" * 95)
    for r in jsonl_reports:
        prims_str = ", ".join(f"{k}:{v}" for k, v in r["primitives_distribution"].items()) if r["primitives_distribution"] else "<Empty>"
        replay_flag = f"🚨 {r['replay_tag_count']}" if r["replay_tag_count"] > 0 else "0 (安全)"
        print(f"{r['file_name']:<40} {r['total_lines']:<8} {r['size_mb']:<10.3f} {prims_str:<22} {replay_flag:<10}")

    if other_files:
        print("\n[📦 其他格式资产文件 (Parquet / Bin 等)]:")
        for o in other_files:
            print(f"  • {o['file_name']:<35} ({o['size_mb']:.2f} MB, Hash: {o['sha256_prefix']})")

    # 2. 深度契约切片
    print("\n[🔬 关键数据集契约结构切片]:")
    for r in jsonl_reports:
        if r["total_lines"] > 2:
            s = r["schema_sample"]
            print(f"\n  • 数据集: [{r['file_name']}] (有效样本: {r['valid_json_count']} 条)")
            print(f"    ├─ 首行 Sample ID    : {s.get('sample_id')}")
            print(f"    ├─ 核心主键原语     : {s.get('primitive')}")
            print(f"    ├─ 包含双正交目标得分: DualScore={r['has_dual_target_scores_ratio']} (Impact: {s.get('has_impact_head_target')}, Sens: {s.get('has_sens_head_target')})")
            print(f"    └─ 字段键名集合     : {s.get('keys')}")

    print("\n" + "=" * 85)
    print(" [✓] 资产物理探查完毕，零数据写入，零破坏操作。")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
