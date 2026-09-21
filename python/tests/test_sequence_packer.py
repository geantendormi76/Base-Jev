import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-1.7B-Base"

import pytest
from transformers import AutoTokenizer
from contracts import (
    BatchDecisionRequest,
    BooleanQuestion,
    CandidateOption,
    ChoiceQuestion,
    ScoreQuestion,
    TypedStateSnapshot,
)
from sequence_packer import (
    CAND_ANCHOR_TOKEN,
    QUERY_ANCHOR_TOKEN,
    SequencePacker,
)


def test_sequence_packer_anchors_and_causal_order():
    """验证因果打包：特殊标记原子化，候选先于查询，锚点下标完全精准匹配"""
    if not MODEL_DIR.exists():
        pytest.skip(f"本地 {MODEL_DIR} 资产未就绪，跳过测试")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    packer = SequencePacker(tokenizer)

    # 构造同时包含 Choice / Boolean / Score 的复合请求
    req = BatchDecisionRequest(
        state=TypedStateSnapshot(
            module="router",
            ast_node="PacketHandler",
            scope="filter",
            attributes=[("traffic_type", "WAN")],
        ),
        choices=[
            ChoiceQuestion(
                id="q_route",
                query="Select forward interface",
                candidates=[
                    CandidateOption(index=0, text="eth0_direct"),
                    CandidateOption(index=1, text="eth1_tunnel"),
                    CandidateOption(index=2, text="drop_blackhole"),
                ],
            )
        ],
        booleans=[BooleanQuestion(id="q_alert", query="Trigger intrusion alert?")],
        scores=[ScoreQuestion(id="q_severity", query="Threat score", active_levels=5)],
    )

    packed = packer.pack_request(req)

    # 1. 验证张量形状
    assert packed.input_ids.shape[0] == 1
    assert packed.attention_mask.shape == packed.input_ids.shape

    # 2. 验证 Choice 锚点精度
    assert len(packed.choice_slices) == 1
    c_slice = packed.choice_slices[0]
    assert len(c_slice.candidate_token_indices) == 3
    
    input_ids_list = packed.input_ids[0].tolist()
    # 断言每个记录的 candidate 索引确实对应 <|cand_anchor|>
    for cand_idx in c_slice.candidate_token_indices:
        assert input_ids_list[cand_idx] == packer.cand_anchor_id

    # 断言 query 索引确实对应 <|query_anchor|>
    assert input_ids_list[c_slice.query_token_idx] == packer.query_anchor_id

    # 3. 验证因果时序定理：所有候选锚点必须严格位于当前查询锚点的前面！
    for cand_idx in c_slice.candidate_token_indices:
        assert cand_idx < c_slice.query_token_idx

    # 4. 验证 Boolean 与 Score 锚点
    assert len(packed.boolean_slices) == 1
    assert input_ids_list[packed.boolean_slices[0].query_token_idx] == packer.query_anchor_id

    assert len(packed.score_slices) == 1
    assert input_ids_list[packed.score_slices[0].query_token_idx] == packer.query_anchor_id

