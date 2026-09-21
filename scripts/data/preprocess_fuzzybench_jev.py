import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "python" / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import pyarrow.parquet as pq

DATA_DIR = PROJECT_ROOT / "data" / "fuzzy_bench"
OUTPUT_DIR = PROJECT_ROOT / "data"

@dataclass
class FuzzyTaskMeta:
    task_id: str
    spec_text: str
    candidates: List[str]
    is_valid_choice: bool
    rejection_reason: str = ""


def analyze_task_codomain(spec_text: str, distinct_outputs: List[str]) -> FuzzyTaskMeta:
    """
    纯泛化值域分析引擎：先定性（过滤长文本生成），后定量（过滤基数异常）
    """
    spec_hash = hashlib.md5(spec_text.encode("utf-8")).hexdigest()[:8]
    task_id = f"TASK-{spec_hash}"

    # 1. 优先定性：过滤长文本/多行自由生成任务 (任何一个输出大于 3 词或超过 30 字符)
    for out in distinct_outputs:
        s = str(out).strip()
        words = s.split()
        if len(words) > 3 or len(s) > 30 or "\n" in s or "\r" in s:
            return FuzzyTaskMeta(task_id, spec_text, [], False, f"Output too long or multi-line: '{s[:25]}...'")

    # 2. 定量检查：过滤候选过少任务 (少于 2 个选项无法构成选择)
    if len(distinct_outputs) < 2:
        return FuzzyTaskMeta(task_id, spec_text, [], False, "Cardinality < 2 (insufficient candidates)")

    # 3. 定量检查：过滤候选过多或开放集任务 (大于 10 个选项不属于标准短决策)
    if len(distinct_outputs) > 10:
        return FuzzyTaskMeta(task_id, spec_text, [], False, "Cardinality > 10 (too high)")

    # 规范化：按字典序对候选排序，保证跨样本索引绝对一致
    sorted_cands = sorted(list(set(str(o).strip() for o in distinct_outputs)))
    return FuzzyTaskMeta(task_id, spec_text, sorted_cands, True, "")


def convert_row_to_jev_choice_sample(
    sample_idx: int,
    meta: FuzzyTaskMeta,
    input_text: str,
    target_output: str,
    label_smoothing: float = 0.05,
) -> Dict:
    """
    将原生 FuzzyBench 数据行转换为符合 Jev 训练母机 contracts 规范的强类型样本字典
    """
    target_clean = str(target_output).strip()
    k = len(meta.candidates)

    if target_clean in meta.candidates:
        target_idx = meta.candidates.index(target_clean)
    else:
        target_idx = 0

    # 认识论软标签构建：平滑独热概率，避免过度自信
    if label_smoothing > 0.0 and k > 1:
        smooth_val = label_smoothing / (k - 1)
        target_dist = [round(smooth_val, 5)] * k
        target_dist[target_idx] = round(1.0 - label_smoothing, 5)
    else:
        target_dist = [0.0] * k
        target_dist[target_idx] = 1.0

    # 确保浮点和严格归一为 1.0
    gap = 1.0 - sum(target_dist)
    target_dist[target_idx] = round(target_dist[target_idx] + gap, 5)

    cand_list = [{"index": i, "text": c} for i, c in enumerate(meta.candidates)]
    query_line = meta.spec_text[:120].strip()

    return {
        "sample_id": f"FUZZY-CHOICE-{sample_idx:06d}",
        "provenance": "SYNTHETIC",
        "category": "POSITIVE",
        "primitive": "Choice",
        "framework": "FuzzyBench",
        "host": "fuzzybench.foundation",
        "method": "DECIDE",
        "path": f"/fuzzy/{meta.task_id.lower()}",
        "extracted_params": [],
        "code_slice": input_text.strip()[:350],
        "query": f"根据模糊规约判决候选分支: {query_line}",
        "candidates": cand_list,
        "is_ambiguous": 0.0,
        "ast_node_type": "CallExpression",
        "target_distribution": target_dist,
    }


def process_split(
    parquet_path: Path,
    output_jsonl_path: Path,
    max_samples: int = 50000,
) -> int:
    print(f"\n[*] 正在读取并聚类任务值域: {parquet_path.name} ...")
    table = pq.read_table(parquet_path)
    
    specs = table.column("spec").to_pylist()
    inputs = table.column("input").to_pylist()
    outputs = table.column("output").to_pylist()
    total_rows = len(specs)

    # 阶段 1：全局聚类每个 spec 的合法离散值域集合
    spec_outputs_map: Dict[str, Set[str]] = {}
    for spec, out in zip(specs, outputs):
        s_text = str(spec).strip()
        o_text = str(out).strip()
        if s_text not in spec_outputs_map:
            spec_outputs_map[s_text] = set()
        spec_outputs_map[s_text].add(o_text)

    # 阶段 2：判定各任务合法性
    spec_meta_map: Dict[str, FuzzyTaskMeta] = {}
    valid_task_count = 0
    for spec, out_set in spec_outputs_map.items():
        meta = analyze_task_codomain(spec, list(out_set))
        spec_meta_map[spec] = meta
        if meta.is_valid_choice:
            valid_task_count += 1

    print(f"  ├─ 发现唯一模糊任务规格 (Specs) : {len(spec_outputs_map)} 个")
    print(f"  └─ 确认为纯正 Choice 决策任务   : {valid_task_count} 个")

    # 阶段 3：流式转换并落盘
    print(f"[*] 正在将高纯度 Choice 样本转换为 Jev 契约格式 (上限 {max_samples:,} 条)...")
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    written_count = 0
    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for idx in range(total_rows):
            spec = str(specs[idx]).strip()
            meta = spec_meta_map.get(spec)
            if not meta or not meta.is_valid_choice:
                continue

            raw_input_list = inputs[idx]
            input_text = ""
            if isinstance(raw_input_list, list) and len(raw_input_list) > 0:
                first_item = raw_input_list[0]
                if isinstance(first_item, dict):
                    input_text = first_item.get("text", "")
                else:
                    input_text = str(first_item)
            else:
                input_text = str(raw_input_list)

            if not input_text.strip():
                continue

            sample_dict = convert_row_to_jev_choice_sample(
                sample_idx=written_count + 1,
                meta=meta,
                input_text=input_text,
                target_output=str(outputs[idx]).strip(),
                label_smoothing=0.05,
            )
            f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
            written_count += 1

            if written_count >= max_samples:
                break

    print(f"  └─ [✓] 资产已固化: {output_jsonl_path.name} (共写入 {written_count:,} 条纯净样本)")
    return written_count


def main():
    print("=" * 75)
    print(" 🏭 base-jev: FuzzyBench 模糊通识数据集提纯与 Jev 契约转换流水线")
    print(f" 📂 原始 Parquet 源目录: {DATA_DIR}")
    print("=" * 75)

    train_parquet = DATA_DIR / "train-00000-of-00001.parquet"
    val_parquet = DATA_DIR / "val-00000-of-00001.parquet"

    if not train_parquet.is_file() or not val_parquet.is_file():
        raise FileNotFoundError(f"缺少必要的 parquet 数据文件: {DATA_DIR}")

    out_val = OUTPUT_DIR / "fuzzybench_jev_val_5k.jsonl"
    process_split(val_parquet, out_val, max_samples=5000)

    out_train = OUTPUT_DIR / "fuzzybench_jev_train_50k.jsonl"
    process_split(train_parquet, out_train, max_samples=50000)

    print("\n" + "=" * 75)
    print(" 🎉 FuzzyBench 模糊通识通配样本提纯全流程圆满竣工！")
    print(f" 📦 通识训练集 : {out_train.resolve()} (50,000 条)")
    print(f" 📦 通识验证集 : {out_val.resolve()} (5,000 条)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

