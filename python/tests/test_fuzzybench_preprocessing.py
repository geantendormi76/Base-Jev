import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = PROJECT_ROOT / "python" / "packages" / "core" / "src"
PIPELINE_SRC = PROJECT_ROOT / "python" / "scripts" / "data"

for p in [PYTHON_SRC, PIPELINE_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest
from contracts import CandidateOption, ChoiceQuestion, TypedStateSnapshot, BatchDecisionRequest
from sequence_packer import SequencePacker
from transformers import AutoTokenizer

# 导入即将实现的纯泛化值域提纯引擎核心函数
from preprocess_fuzzybench_jev import (
    analyze_task_codomain,
    convert_row_to_jev_choice_sample,
    FuzzyTaskMeta,
)

MODEL_06B_DIR = PROJECT_ROOT / "models" / "Qwen3-0.6B-Base"


def test_codomain_grouping_pure_generalization():
    """验证泛化定理：利用离散值域集合自动推导候选分支，杜绝正则特判"""
    # 模拟任务 A: 经典 3 分类词义消歧
    outputs_valid = ["financial_institution", "river_edge", "rely_on"]
    meta_valid = analyze_task_codomain("Disambiguate bank...", outputs_valid)

    assert meta_valid.is_valid_choice is True
    assert len(meta_valid.candidates) == 3
    assert meta_valid.candidates == ["financial_institution", "rely_on", "river_edge"]  # 字典序确定性

    # 模拟任务 B: 自由长文本生成 (输出大于 3 词且长达 50 字符，必须拒绝)
    outputs_generative = ["This is a long repaired JSON string that should be rejected by Jev."]
    meta_gen = analyze_task_codomain("Repair JSON...", outputs_generative)
    assert meta_gen.is_valid_choice is False
    assert "length" in meta_gen.rejection_reason.lower() or "too long" in meta_gen.rejection_reason.lower()

    # 模拟任务 C: 开放集无限输出 (|Y| > 10，如生成自由数字或随机名称，必须拒绝)
    outputs_high_cardinality = [f"tag_{i}" for i in range(15)]
    meta_high = analyze_task_codomain("Generate unique ID...", outputs_high_cardinality)
    assert meta_high.is_valid_choice is False
    assert "cardinality" in meta_high.rejection_reason.lower()


def test_convert_row_to_jev_choice_contract():
    """验证数据契约：转换出的样本必须满足 Choice 原语与概率归一化公理"""
    meta = FuzzyTaskMeta(
        task_id="TASK-WSD-01",
        spec_text="Disambiguate word bank into one of three senses",
        candidates=["financial_institution", "rely_on", "river_edge"],
        is_valid_choice=True,
    )

    sample = convert_row_to_jev_choice_sample(
        sample_idx=1,
        meta=meta,
        input_text="I deposited the check at the bank.",
        target_output="financial_institution",
        label_smoothing=0.05,
    )

    # 1. 验证契约基本字段
    assert sample["primitive"] == "Choice"
    assert sample["sample_id"] == "FUZZY-CHOICE-000001"
    assert len(sample["candidates"]) == 3
    assert sample["candidates"][0]["text"] == "financial_institution"

    # 2. 验证软概率标签代数约束 (经过 label_smoothing 平滑，严格防过拟合)
    dist = sample["target_distribution"]
    assert len(dist) == 3
    assert pytest.approx(sum(dist), abs=1e-5) == 1.0
    # 目标项占据大头 (约 0.95)，其余平分 0.05
    assert dist[0] > 0.90
    assert dist[1] > 0.01 and dist[2] > 0.01


def test_converted_sample_sequence_packing_invariants():
    """验证序列打包：转换后的 FuzzyBench 样本可被 SequencePacker 零报错装配并锁定锚点"""
    if not MODEL_06B_DIR.is_dir():
        pytest.skip("本地 0.6B 基模未就绪，跳过物理 Tokenizer 测试")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_06B_DIR), trust_remote_code=True)
    packer = SequencePacker(tokenizer)

    meta = FuzzyTaskMeta(
        task_id="TASK-WSD-01",
        spec_text="Classify relation as causal or correlational",
        candidates=["causal", "correlational"],
        is_valid_choice=True,
    )

    sample = convert_row_to_jev_choice_sample(
        sample_idx=42,
        meta=meta,
        input_text="Smoking causes lung cancer in long term studies.",
        target_output="causal",
    )

    # 模拟接入 BatchDecisionRequest
    cand_options = [CandidateOption(index=c["index"], text=c["text"]) for c in sample["candidates"]]
    req = BatchDecisionRequest(
        state=TypedStateSnapshot(
            module="FuzzyBench",
            ast_node="TaskSpec",
            scope=sample["path"],
            attributes=[("context", sample["code_slice"])],
        ),
        choices=[ChoiceQuestion(id=sample["sample_id"], query=sample["query"], candidates=cand_options)],
    )

    packed = packer.pack_request(req)

    # 1. 断言 Choice 锚点存在
    assert len(packed.choice_slices) == 1
    c_slice = packed.choice_slices[0]
    assert len(c_slice.candidate_token_indices) == 2
    # 2. 候选锚点必须在 Query 锚点之前 (因果时序定理)
    assert c_slice.candidate_token_indices[0] < c_slice.query_token_idx
    assert c_slice.candidate_token_indices[1] < c_slice.query_token_idx


