from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path

TRAIN_PATH = Path("data/distilled_invar_train.jsonl")
PRED_PATH = Path("C:/dev/Invar/tmp/base_jev_predictions_1221.jsonl")
GOLD_PATH = Path("C:/dev/Invar/data/targets/ikuai8.com/gold_set_annotated_v1.jsonl")
MANIFEST_PATH = Path("C:/dev/Invar/tmp/triage_phase1_output/triage_run_manifest_v2.json")

def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    sorted_v = sorted(vals)
    k = (len(sorted_v) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)

def main():
    print("=" * 80)
    print(" 🔬 数据与分值全景显微镜 (Train Set vs Predictions vs Gold Set)")
    print("=" * 80)

    # 1. 探查训练集
    print("\n[1] 正在解构训练集: data/distilled_invar_train.jsonl")
    primitives = Counter()
    train_imp = []
    train_sens = []
    total_train = 0

    if TRAIN_PATH.is_file():
        with TRAIN_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_train += 1
                    row = json.loads(line)
                    prim = row.get("primitive", "UNKNOWN")
                    primitives[prim] += 1
                    
                    # 安全获取 target_scores (防止 null)
                    ts = row.get("target_scores") or {}
                    imp_target = ts.get("impact")
                    sens_target = ts.get("sensitivity")
                    
                    if isinstance(imp_target, dict) and "expected_score" in imp_target:
                        train_imp.append(float(imp_target["expected_score"]))
                    if isinstance(sens_target, dict) and "expected_score" in sens_target:
                        train_sens.append(float(sens_target["expected_score"]))

        print(f"  ├─ 样本总量           : {total_train} 条")
        print(f"  ├─ 原语构成 (Primitives): {dict(primitives)}")
        if train_imp:
            print(f"  ├─ Impact 标签分值    : Min={min(train_imp):.2f}, Max={max(train_imp):.2f}, Mean={sum(train_imp)/len(train_imp):.2f}")
            print(f"  │  └─ 分值 >= 3.5 样本数: {sum(1 for x in train_imp if x >= 3.5)} 条 ({sum(1 for x in train_imp if x >= 3.5)/len(train_imp)*100:.1f}%)")
            print(f"  └─ Sens   标签分值    : Min={min(train_sens):.2f}, Max={max(train_sens):.2f}, Mean={sum(train_sens)/len(train_sens):.2f}")
            print(f"     └─ 分值 >= 3.5 样本数: {sum(1 for x in train_sens if x >= 3.5)} 条 ({sum(1 for x in train_sens if x >= 3.5)/len(train_sens)*100:.1f}%)")
        else:
            print("  └─ [!] 警告：未解析到任何合法的 expected_score 标签！")

    # 2. 探查预测集
    print("\n[2] 正在解构预测集: C:/dev/Invar/tmp/base_jev_predictions_1221.jsonl")
    pred_imp = []
    pred_sens = []
    if PRED_PATH.is_file():
        with PRED_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pred_imp.append(float(row.get("impact_score", 1.0)))
                    pred_sens.append(float(row.get("sensitivity_score", 1.0)))

        print(f"  ├─ 预测总量           : {len(pred_imp)} 个端点")
        print(f"  ├─ Impact 预测分布    : Min={min(pred_imp):.2f}, P50={percentile(pred_imp, 0.5):.2f}, P95={percentile(pred_imp, 0.95):.2f}, Max={max(pred_imp):.2f}, Mean={sum(pred_imp)/len(pred_imp):.2f}")
        print(f"  ├─ Sens   预测分布    : Min={min(pred_sens):.2f}, P50={percentile(pred_sens, 0.5):.2f}, P95={percentile(pred_sens, 0.95):.2f}, Max={max(pred_sens):.2f}, Mean={sum(pred_sens)/len(pred_sens):.2f}")
        print(f"  └─ 模型认为高危(>=3.5): Impact={sum(1 for x in pred_imp if x >= 3.5)} 个 | Sensitivity={sum(1 for x in pred_sens if x >= 3.5)} 个")

    # 3. 探查黄金集真值卡片
    print("\n[3] 正在解构黄金集真值: C:/dev/Invar/data/targets/ikuai8.com/gold_set_annotated_v1.jsonl")
    if GOLD_PATH.is_file():
        gold_cards = []
        with GOLD_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gold_cards.append(json.loads(line))
        gt_high_cards = []
        for c in gold_cards:
            anno = c.get("annotation", {})
            g_i = anno.get("ground_truth_impact", 3)
            g_s = anno.get("ground_truth_sensitivity", 3)
            if g_i >= 4 or g_s >= 4:
                gt_high_cards.append((c.get("card_id"), c.get("method"), c.get("path"), g_i, g_s))

        print(f"  ├─ 黄金卡片总量       : {len(gold_cards)} 张")
        print(f"  ├─ 真实高危卡片总数   : {len(gt_high_cards)} 张")
        print("  └─ 真实高危样本切片清单:")
        for cid, m, p, gi, gs in gt_high_cards:
            print(f"     • [{cid}] {m:6s} {p:45s} (GT: Impact={gi}, Sens={gs})")

    # 4. 探查比较器 Manifest 哈希封签
    print("\n[4] 正在检查比较器血缘封签: triage_run_manifest_v2.json")
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        inputs = manifest.get("inputs", {})
        has_hash = "neural_jsonl_sha256" in inputs
        print(f"  ├─ 包含 neural_jsonl_sha256: {'✅ 已焊入 (' + inputs.get('neural_jsonl_sha256', '')[:16] + '...)' if has_hash else '❌ 缺失'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
