# -*- coding: utf-8 -*-
"""
repair_distiller_imports.py
规范化修复 C:/dev/distiller 脚本的导入顺序与类型注解环境
"""
from pathlib import Path

PIPELINE_PATH = Path("C:/dev/distiller/scripts/pipeline/run_distillation_pipeline.py")
CURATOR_PATH = Path("C:/dev/distiller/src-tauri/python/src/curator.py")

def fix_pipeline():
    content = PIPELINE_PATH.read_text(encoding="utf-8")
    
    # 提取并移除头部错位的 load_excluded_surfaces 代码块
    load_func_marker = "def load_excluded_surfaces(file_path_str:"
    if content.startswith(load_func_marker):
        end_marker = "return excluded\n\n"
        idx = content.find(end_marker)
        if idx != -1:
            content = content[idx + len(end_marker):]

    # 清理并规范化顶部导入
    top_block = '''from __future__ import annotations
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

def load_excluded_surfaces(file_path_str: Optional[str]) -> Set[str]:
    excluded = set()
    if not file_path_str:
        return excluded
    p = Path(file_path_str)
    if not p.is_file():
        return excluded
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for m in data.get("key_mappings", []):
                if "surface_id" in m:
                    excluded.add(m["surface_id"])
        except Exception:
            pass
    elif p.suffix.lower() == ".jsonl":
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    sid = row.get("surface_id")
                    if sid:
                        excluded.add(sid)
                    m = str(row.get("method", "")).upper()
                    h = str(row.get("host", "")).lower()
                    path_str = str(row.get("path", ""))
                    if m and path_str:
                        excluded.add(f"{m}|{path_str}")
                    if m and h and path_str:
                        excluded.add(f"{m}|{h}|{path_str}")
                        excluded.add(f"{h}|{m}|{path_str}")
        except Exception:
            pass
    return excluded

'''
    # 移除现有重复的 sys/Path 导入前缀
    lines = content.splitlines()
    clean_lines = []
    skip = True
    for line in lines:
        if skip and (line.startswith("import ") or line.startswith("from ") or not line.strip()):
            continue
        skip = False
        clean_lines.append(line)

    final_content = top_block + "\n".join(clean_lines)
    PIPELINE_PATH.write_text(final_content, encoding="utf-8")
    print("[✓] run_distillation_pipeline.py 导入顺序修复完成！")

def fix_curator():
    content = CURATOR_PATH.read_text(encoding="utf-8")
    # 确保 curator.py 头部带有 from __future__ import annotations 并包含 Set
    if "from __future__ import annotations" not in content:
        content = "from __future__ import annotations\n" + content
    
    # 替换 typing 导入包含 Set
    content = re_replace_typing(content)
    CURATOR_PATH.write_text(content, encoding="utf-8")
    print("[✓] curator.py 导入与 Set 类型支持修复完成！")

def re_replace_typing(text: str) -> str:
    import re
    return re.sub(
        r"from typing import ([^\n]+)",
        r"from typing import Any, Dict, List, Optional, Set, Tuple, Union",
        text
    )

if __name__ == "__main__":
    fix_pipeline()
    fix_curator()
