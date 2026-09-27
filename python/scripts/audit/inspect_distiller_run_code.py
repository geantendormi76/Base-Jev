# -*- coding: utf-8 -*-
"""
inspect_distiller_run_code.py
深度解构 run_distillation_pipeline.py 调度逻辑与本地教师模型服务存活检测
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

DISTILLER_PIPELINE = Path("C:/dev/distiller/scripts/pipeline/run_distillation_pipeline.py")

def check_endpoint_health(endpoint: str = "http://127.0.0.1:8080/v1") -> dict:
    url = f"{endpoint.rstrip('/')}/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="ignore")
            return {"alive": status == 200, "status_code": status, "preview": body[:150]}
    except Exception as exc:
        return {"alive": False, "error": str(exc)}

def main():
    print("=" * 85)
    print(" 🔬 正在解构 run_distillation_pipeline.py 全流程代码逻辑...")
    print("=" * 85)

    if not DISTILLER_PIPELINE.is_file():
        print(f"[❌] 文件不存在: {DISTILLER_PIPELINE}")
        return

    content = DISTILLER_PIPELINE.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    # 1. 检测本地 27B 教师模型服务存活状态
    print("\n[1] 正在探测本地教师模型推理端口 (http://127.0.0.1:8080/v1)...")
    health = check_endpoint_health()
    if health["alive"]:
        print(f"  └─ [✓] 教师模型服务在线！HTTP {health['status_code']} | 响应: {health['preview']}")
    else:
        print(f"  └─ [!] 教师模型服务离线或未启动: {health.get('error')}")

    # 2. 打印流水线的核心阶段逻辑
    print(f"\n[2] 脚本全景 ({len(lines)} 行) 核心流程切片:")
    in_main = False
    for idx, line in enumerate(lines, 1):
        if "def main" in line:
            in_main = True
        if in_main:
            # 打印 main 函数及关键管道步骤
            if any(k in line for k in ["curator", "synthesizer", "annotator", "write", "sync", "stage", "print("]) or idx > len(lines) - 40:
                print(f"  {idx:3d} | {line}")

    print("\n" + "=" * 85)
    print(" [✓] 解构探查完毕，零数据修改。")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
