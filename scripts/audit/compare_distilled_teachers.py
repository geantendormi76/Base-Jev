from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

LOCAL_TRAIN_PATH = Path("data/Local_distilled/distilled_invar_train.jsonl")
API_TRAIN_PATH = Path("data/API_distilled/distilled_invar_train.jsonl")

def load_dataset_map(path: Path) -> Dict[str, Dict[str, Any]]:
    mapping = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sid = row.get("sample_id")
                if sid:
                    mapping[sid] = row
    return mapping

def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)

def main():
    print("=" * 85)
    print(" 🔬 蒸馏双师终审对决: Local Qwen-27B vs Cloud Gemini Flash")
    print("=" * 85)

    local_map = load_dataset_map(LOCAL_TRAIN_PATH)
    api_map = load_dataset_map(API_TRAIN_PATH)

    print(f"[*] 本地 27B 训练集样本数: {len(local_map)} 条 ({LOCAL_TRAIN_PATH})")
    print(f"[*] 云端 API 训练集样本数: {len(api_map)} 条 ({API_TRAIN_PATH})")

    common_ids = sorted(list(set(local_map.keys()) & set(api_map.keys())))
    print(f"[*] 共同评估重合样本数   : {len(common_ids)} 条 (完全同构 AST 切片)")
    print("-" * 85)

    local_impacts, local_sens = [], []
    api_impacts, api_sens = [], []

    impact_diffs = []
    sens_diffs = []

    # 共识判定 (高危判定阈值 >= 3.5)
    both_high_impact = 0
    both_low_impact = 0
    dispute_impact = 0

    both_high_sens = 0
    both_low_sens = 0
    dispute_sens = 0

    discrepancy_cases = []

    for sid in common_ids:
        l_row = local_map[sid]
        a_row = api_map[sid]

        l_ts = l_row.get("target_scores") or {}
        a_ts = a_row.get("target_scores") or {}

        l_imp = l_ts.get("impact", {}).get("expected_score")
        l_se = l_ts.get("sensitivity", {}).get("expected_score")

        a_imp = a_ts.get("impact", {}).get("expected_score")
        a_se = a_ts.get("sensitivity", {}).get("expected_score")

        if l_imp is not None and a_imp is not None:
            local_impacts.append(l_imp)
            api_impacts.append(a_imp)
            diff_i = abs(l_imp - a_imp)
            impact_diffs.append(diff_i)

            l_hi = l_imp >= 3.5
            a_hi = a_imp >= 3.5
            if l_hi and a_hi:
                both_high_impact += 1
            elif not l_hi and not a_hi:
                both_low_impact += 1
            else:
                dispute_impact += 1

        if l_se is not None and a_se is not None:
            local_sens.append(l_se)
            api_sens.append(a_se)
            diff_s = abs(l_se - a_se)
            sens_diffs.append(diff_s)

            l_hs = l_se >= 3.5
            a_hs = a_se >= 3.5
            if l_hs and a_hs:
                both_high_sens += 1
            elif not l_hs and not a_hs:
                both_low_sens += 1
            else:
                dispute_sens += 1

        if l_imp is not None and a_imp is not None:
            if abs(l_imp - a_imp) >= 1.5 or abs(l_se - a_se) >= 1.5:
                discrepancy_cases.append({
                    "sample_id": sid,
                    "route": f"{l_row.get('method')} {l_row.get('path')}",
                    "local": f"I={l_imp:.2f}, S={l_se:.2f}",
                    "api": f"I={a_imp:.2f}, S={a_se:.2f}",
                    "code": l_row.get("code_slice", "")[:80]
                })

    def print_stat(name: str, vals: List[float]):
        m = sum(vals) / len(vals)
        high_cnt = sum(1 for x in vals if x >= 3.5)
        print(f"  • {name:20s}: Min={min(vals):.2f}, P50={percentile(vals, 0.5):.2f}, Max={max(vals):.2f}, Mean={m:.2f} | 高危(>=3.5)={high_cnt} ({high_cnt/len(vals)*100:.1f}%)")

    print("\n[1] 破坏力 (Impact) 宏观认知分布对照:")
    print_stat("Local 27B Impact", local_impacts)
    print_stat("Cloud API Impact", api_impacts)
    print(f"  ├─ 双方平均分值绝对误差 (MAE): {sum(impact_diffs)/len(impact_diffs):.3f} 档")
    print(f"  └─ 高危共识率: 双师共识判定高危={both_high_impact} | 双师共识低危={both_low_impact} | 产生分歧={dispute_impact} ({dispute_impact/len(common_ids)*100:.1f}%)")

    print("\n[2] 敏感度 (Sensitivity) 宏观认知分布对照:")
    print_stat("Local 27B Sensitivity", local_sens)
    print_stat("Cloud API Sensitivity", api_sens)
    print(f"  ├─ 双方平均分值绝对误差 (MAE): {sum(sens_diffs)/len(sens_diffs):.3f} 档")
    print(f"  └─ 高危共识率: 双师共识判定高危={both_high_sens} | 双师共识低危={both_low_sens} | 产生分歧={dispute_sens} ({dispute_sens/len(common_ids)*100:.1f}%)")

    print("\n" + "=" * 85)
    print(" [3] 典型高分歧案例抽检 (Discrepancy Samples):")
    print("=" * 85)
    for idx, c in enumerate(discrepancy_cases[:6], 1):
        print(f"  {idx}. [{c['sample_id']}] {c['route']}")
        print(f"     ├─ 本地 27B 判定 : {c['local']}")
        print(f"     ├─ 云端 API 判定 : {c['api']}")
        print(f"     └─ 代码切片前瞻 : {c['code']}...")

    print("\n" + "=" * 85)
    print(" 🎯 综合比对结论就绪！")
    print("=" * 85)

if __name__ == "__main__":
    main()
