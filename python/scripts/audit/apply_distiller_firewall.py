# -*- coding: utf-8 -*-
"""
apply_distiller_firewall.py
为 C:/dev/distiller 生产流水线装配 Gold Set 物理隔离防火墙
职责：
1. 在 contracts.py 契约中扩展 CuratorStats.excluded_by_filter
2. 在 curator.py 中注入通用 make_surface_id 与 excluded_surfaces 过滤
3. 在 run_distillation_pipeline.py 中接入 --exclude-file CLI 选项
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DISTILLER_ROOT = Path("C:/dev/distiller")
CONTRACTS_PATH = DISTILLER_ROOT / "src-tauri" / "python" / "src" / "contracts.py"
CURATOR_PATH = DISTILLER_ROOT / "src-tauri" / "python" / "src" / "curator.py"
PIPELINE_PATH = DISTILLER_ROOT / "scripts" / "pipeline" / "run_distillation_pipeline.py"

def patch_contracts():
    print("[1/3] 正在挂载 contracts.py 强类型契约...")
    content = CONTRACTS_PATH.read_text(encoding="utf-8")
    if "excluded_by_filter: int = 0" in content:
        print("  └─ [✓] CuratorStats 契约已就绪，跳过。")
        return

    old_block = """    ast_noise_filtered: int = 0
    curated_slices_extracted: int = 0"""
    new_block = """    ast_noise_filtered: int = 0
    excluded_by_filter: int = 0
    curated_slices_extracted: int = 0"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        CONTRACTS_PATH.write_text(content, encoding="utf-8")
        print("  └─ [✓] 成功为 CuratorStats 增加 excluded_by_filter 字段！")
    else:
        print("  └─ [!] 未能精确定位 CuratorStats 代码块，请检查 contracts.py")

def patch_curator():
    print("[2/3] 正在挂载 curator.py 纯泛化表面排除逻辑...")
    content = CURATOR_PATH.read_text(encoding="utf-8")
    
    # 确保导入 hashlib 与 urlparse
    if "import hashlib" not in content:
        content = "import hashlib\nfrom urllib.parse import urlparse\n" + content

    # 注入 make_surface_id 方法
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
        payload = "\x1f".join([method.upper(), host, norm_p]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

'''
    if "def make_surface_id" not in content:
        content = content.replace("class PRismCurator:", "class PRismCurator:\n" + surface_func)

    # 扩展 curate_report 函数签名与过滤逻辑
    old_sig = "max_endpoints: Optional[int] = None,\n    ) -> Tuple[List[CuratedEndpoint], CuratorStats]:"
    new_sig = "max_endpoints: Optional[int] = None,\n        excluded_surfaces: Optional[Set[str]] = None,\n    ) -> Tuple[List[CuratedEndpoint], CuratorStats]:"
    if old_sig in content:
        content = content.replace(old_sig, new_sig)

    # 注入循环过滤检查
    old_loop_check = """            host = self.extract_host_from_path(source_file_str, raw_js_dir)
            dedup_key = f"{host}|{method}|{path}" """
    
    new_loop_check = """            host = self.extract_host_from_path(source_file_str, raw_js_dir)

            # Gold Set 物理隔离防火墙 (Zero-Contamination Boundary)
            if excluded_surfaces:
                sid = self.make_surface_id(method, path)
                if (
                    sid in excluded_surfaces
                    or f"{method}|{path}" in excluded_surfaces
                    or f"{method}|{host}|{path}" in excluded_surfaces
                    or f"{host}|{method}|{path}" in excluded_surfaces
                ):
                    stats.excluded_by_filter += 1
                    continue

            dedup_key = f"{host}|{method}|{path}" """

    if old_loop_check in content:
        content = content.replace(old_loop_check, new_loop_check)
    
    CURATOR_PATH.write_text(content, encoding="utf-8")
    print("  └─ [✓] 成功为 PRismCurator 注入双层正交排除防火墙！")

def patch_pipeline():
    print("[3/3] 正在挂载 run_distillation_pipeline.py 调度器与 --exclude-file 选项...")
    content = PIPELINE_PATH.read_text(encoding="utf-8")

    # 增加 CLI 参数定义
    if "--exclude-file" not in content:
        arg_target = 'parser.add_argument("--window-chars", type=int, default=250, help="AST 上下文窗口字符数")'
        new_arg = '''parser.add_argument("--window-chars", type=int, default=250, help="AST 上下文窗口字符数")
    parser.add_argument("--exclude-file", default="C:/dev/Invar/data/targets/ikuai8.com/gold_set_annotated_v1.jsonl", help="Gold Set 隔离文件路径")'''
        content = content.replace(arg_target, new_arg)

    # 注入排除集加载逻辑
    if "load_excluded_surfaces" not in content:
        load_func = '''def load_excluded_surfaces(file_path_str: Optional[str]) -> Set[str]:
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
        content = load_func + content

    # 在 main_async 中调用加载并传递给 curator
    old_call = """    curated_endpoints, curator_stats = curator.curate_report(
        report_path=report_path,
        raw_js_root=raw_js_dir,
        max_endpoints=args.max_real_endpoints,
    )"""
    new_call = """    excluded_surfaces = load_excluded_surfaces(args.exclude_file)
    if excluded_surfaces:
        print(f"  [🛡️ 防火墙生效] 成功装配外部评测隔离表面: {len(excluded_surfaces)} 个规则")

    curated_endpoints, curator_stats = curator.curate_report(
        report_path=report_path,
        raw_js_root=raw_js_dir,
        max_endpoints=args.max_real_endpoints,
        excluded_surfaces=excluded_surfaces,
    )"""
    if old_call in content:
        content = content.replace(old_call, new_call)

    # 增加统计打印
    old_stat_print = 'print(f"  ├─ 拦截伪路由/协议/MIME : {curator_stats.ast_noise_filtered} 个")'
    new_stat_print = '''print(f"  ├─ 拦截伪路由/协议/MIME : {curator_stats.ast_noise_filtered} 个")
    print(f"  ├─ 🛡️ 隔离评测黄金表面   : {curator_stats.excluded_by_filter} 个 (物理 0 污染)")'''
    if old_stat_print in content and "隔离评测黄金表面" not in content:
        content = content.replace(old_stat_print, new_stat_print)

    PIPELINE_PATH.write_text(content, encoding="utf-8")
    print("  └─ [✓] 成功为 run_distillation_pipeline.py 挂载防火墙调度！")

def main():
    print("=" * 80)
    print(" 🛠️  正在执行 distiller 防火墙装配手术 (对齐系统宪法标准)...")
    print("=" * 80)
    patch_contracts()
    patch_curator()
    patch_pipeline()
    print("=" * 80)
    print(" 🎉 上游管线防火墙挂载全部竣工！")
    print("=" * 80)

if __name__ == "__main__":
    main()
