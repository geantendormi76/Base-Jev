# -*- coding: utf-8 -*-
"""
repair_pipeline_syntax.py
精准修复 run_distillation_pipeline.py 中的括号失配，彻底消除 SyntaxError
"""
from pathlib import Path

PIPELINE_PATH = Path("C:/dev/distiller/scripts/pipeline/run_distillation_pipeline.py")

def main():
    content = PIPELINE_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()

    new_lines = []
    in_bad_block = False
    skip_until_clean = False

    for idx, line in enumerate(lines):
        # 定位 curate_report 调用段
        if "curated_endpoints, curator_stats = curator.curate_report" in line:
            # 插入干净自洽的调用块
            call_block = """    # 🛡️ Gold Set 物理隔离防火墙装配
    excluded_surfaces = load_excluded_surfaces(args.exclude_file)
    if excluded_surfaces:
        print(f"  [🛡️ 防火墙生效] 成功装配外部评测隔离表面: {len(excluded_surfaces)} 条规则")

    curated_endpoints, curator_stats = curator.curate_report(
        report_path=report_path,
        raw_js_root=raw_js_dir,
        max_endpoints=args.max_real_endpoints,
        excluded_surfaces=excluded_surfaces,
    )"""
            new_lines.append(call_block)
            skip_until_clean = True
            continue

        if skip_until_clean:
            # 跳过破损的旧括号与残留行，直到遇到正常的下一个 print 或 if
            if 'print(f"  ├─ 原始扫描 API 节点数' in line or 'if not curated_endpoints:' in line:
                skip_until_clean = False
                new_lines.append(line)
            continue

        new_lines.append(line)

    fixed_text = "\n".join(new_lines)
    PIPELINE_PATH.write_text(fixed_text, encoding="utf-8")
    print("[✓] run_distillation_pipeline.py 语法结构已精准对齐并修复！")

if __name__ == "__main__":
    main()
