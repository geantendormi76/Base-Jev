# -*- coding: utf-8 -*-
"""
weld_gold_firewall.py
将 Gold Set 隔离防火墙彻底焊死进 C:/dev/distiller 核心模块
"""
from __future__ import annotations

import re
from pathlib import Path

DISTILLER_ROOT = Path("C:/dev/distiller")
CONTRACTS_PATH = DISTILLER_ROOT / "src-tauri" / "python" / "src" / "contracts.py"
CURATOR_PATH = DISTILLER_ROOT / "src-tauri" / "python" / "src" / "curator.py"
PIPELINE_PATH = DISTILLER_ROOT / "scripts" / "pipeline" / "run_distillation_pipeline.py"

# 1. 焊接 contracts.py
def weld_contracts():
    print("[1/3] 焊接 contracts.py: 扩展 CuratorStats 强类型契约...")
    text = CONTRACTS_PATH.read_text(encoding="utf-8")
    if "excluded_by_filter: int = 0" not in text:
        text = text.replace(
            "ast_noise_filtered: int = 0",
            "ast_noise_filtered: int = 0\n    excluded_by_filter: int = 0"
        )
        CONTRACTS_PATH.write_text(text, encoding="utf-8")
        print("  └─ [✓] contracts.py 已焊入 excluded_by_filter 字段")
    else:
        print("  └─ [✓] contracts.py 已经就绪")

# 2. 焊接 curator.py
def weld_curator():
    print("[2/3] 焊接 curator.py: 挂载表面指纹算法与无死角物理拦截网...")
    text = CURATOR_PATH.read_text(encoding="utf-8")
    
    # 顶部补齐 typing 导入
    if "from __future__ import annotations" not in text:
        text = "from __future__ import annotations\n" + text
    
    text = re.sub(
        r"from typing import [^\n]+",
        "from typing import Any, Dict, List, Optional, Set, Tuple, Union",
        text
    )
    if "import hashlib" not in text:
        text = "import hashlib\nfrom urllib.parse import urlparse\n" + text

    # 注入 make_surface_id
    surface_func = '''    @staticmethod
    def make_surface_id(method: str, path: str) -> str:
        parsed = urlparse(path)
        if parsed.scheme and parsed.hostname:
            host = parsed.hostname.lower()
            norm_p = parsed.path or "/"
        else:
            host = "<unknown-host>"
            norm_p = path or "/"
        if parsed.scheme and parsed.hostname and parsed.query:
            norm_p = f"{norm_p}?{parsed.query}"
        payload = "\\x1f".join([method.upper(), host, norm_p]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

'''
    if "def make_surface_id" not in text:
        text = text.replace("class PRismCurator:", "class PRismCurator:\n" + surface_func)

    # 替换 curate_report 的参数签名
    text = text.replace(
        "max_endpoints: Optional[int] = None,\n    ) -> Tuple[List[CuratedEndpoint], CuratorStats]:",
        "max_endpoints: Optional[int] = None,\n        excluded_surfaces: Optional[Set[str]] = None,\n    ) -> Tuple[List[CuratedEndpoint], CuratorStats]:"
    )

    # 替换核心循环去重段，注入三重防御网
    old_loop_target = """            host = self.extract_host_from_path(source_file_str, raw_js_dir)
            dedup_key = f"{host}|{method}|{path}"
            if dedup_key in seen_dedup_keys:
                continue
            seen_dedup_keys.add(dedup_key)"""

    new_loop_target = """            host = self.extract_host_from_path(source_file_str, raw_js_dir)

            # 🛡️ Gold Set 物理隔离防火墙 (Zero-Contamination Boundary)
            if excluded_surfaces:
                sid = self.make_surface_id(method, path)
                clean_p = path.strip()
                clean_p_stripped = clean_p.strip("/")
                is_excluded = (
                    sid in excluded_surfaces
                    or clean_p in excluded_surfaces
                    or f"/{clean_p_stripped}" in excluded_surfaces
                    or clean_p_stripped in excluded_surfaces
                    or f"{method}|{clean_p}" in excluded_surfaces
                    or f"{method}|/{clean_p_stripped}" in excluded_surfaces
                    or f"{method}|{clean_p_stripped}" in excluded_surfaces
                    or f"{method}|{host}|{clean_p}" in excluded_surfaces
                    or f"{method}|{host}|/{clean_p_stripped}" in excluded_surfaces
                    or f"{host}|{method}|{clean_p}" in excluded_surfaces
                )
                if is_excluded:
                    stats.excluded_by_filter += 1
                    continue

            dedup_key = f"{host}|{method}|{path}"
            if dedup_key in seen_dedup_keys:
                continue
            seen_dedup_keys.add(dedup_key)"""

    if old_loop_target in text:
        text = text.replace(old_loop_target, new_loop_target)
        print("  └─ [✓] curator.py 成功焊入三重物理拦截网")
    elif "stats.excluded_by_filter += 1" in text:
        print("  └─ [✓] curator.py 拦截网已存在")
    else:
        print("  └─ [!] 未能精确定位 dedup_key，请核对代码行")

    CURATOR_PATH.write_text(text, encoding="utf-8")

# 3. 焊接 run_distillation_pipeline.py
def weld_pipeline():
    print("[3/3] 焊接 run_distillation_pipeline.py: 绑定参数与全自动特征加载...")
    text = PIPELINE_PATH.read_text(encoding="utf-8")

    # 替换 curate_report 调用
    old_call_pattern = r"curated_endpoints,\s*curator_stats\s*=\s*curator\.curate_report\([^)]+\)"
    new_call_code = """excluded_surfaces = load_excluded_surfaces(args.exclude_file)
    if excluded_surfaces:
        print(f"  [🛡️ 防火墙生效] 成功装配外部评测隔离表面: {len(excluded_surfaces)} 条规则")

    curated_endpoints, curator_stats = curator.curate_report(
        report_path=report_path,
        raw_js_root=raw_js_dir,
        max_endpoints=args.max_real_endpoints,
        excluded_surfaces=excluded_surfaces,
    )"""

    text = re.sub(old_call_pattern, new_call_code, text, count=1)
    
    # 注入打印语句
    old_print = 'print(f"  ├─ 拦截伪路由/协议/MIME : {curator_stats.ast_noise_filtered} 个")'
    new_print = 'print(f"  ├─ 拦截伪路由/协议/MIME : {curator_stats.ast_noise_filtered} 个")\n    print(f"  ├─ 🛡️ 隔离评测黄金表面   : {curator_stats.excluded_by_filter} 个 (物理 0 污染)")'
    if old_print in text and "隔离评测黄金表面" not in text:
        text = text.replace(old_print, new_print)

    PIPELINE_PATH.write_text(text, encoding="utf-8")
    print("  └─ [✓] run_distillation_pipeline.py 调度器焊接完毕！")

if __name__ == "__main__":
    weld_contracts()
    weld_curator()
    weld_pipeline()
    print("🎉 防火墙硬核焊接圆满完成！")
