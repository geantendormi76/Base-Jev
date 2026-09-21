import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-1.7B-Base"

import pytest
import torch
from onnx_export_wrapper import SystemOneOnnxWrapper
from system_one_model import SystemOneModel


def test_onnx_wrapper_forward_and_gather_correctness():
    """验证 ONNX 包装器：Gather 算子正确提取 Query 与 Candidate 特征并完成多原语输出"""
    if not MODEL_DIR.exists():
        pytest.skip(f"本地 {MODEL_DIR} 资产未就绪，跳过物理基模测试")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SystemOneModel.from_pretrained_base(
        model_path_or_id=MODEL_DIR,
        torch_dtype=torch.float16,
    )
    wrapper = SystemOneOnnxWrapper(model.backbone, model.heads)
    wrapper.to(device)
    wrapper.eval()

    # 构造动态输入: 序列长度 16，2 个候选，1 个布尔，1 个打分
    seq_len = 16
    input_ids = torch.randint(100, 1000, (1, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    
    choice_q_pos = torch.tensor([15], dtype=torch.long, device=device)
    choice_cand_pos = torch.tensor([[5, 10]], dtype=torch.long, device=device)
    choice_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=device)
    bool_q_pos = torch.tensor([15], dtype=torch.long, device=device)
    score_q_pos = torch.tensor([15], dtype=torch.long, device=device)

    with torch.no_grad():
        c_probs, b_probs, s_dist, s_exp = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            choice_q_pos=choice_q_pos,
            choice_cand_pos=choice_cand_pos,
            choice_mask=choice_mask,
            bool_q_pos=bool_q_pos,
            score_q_pos=score_q_pos,
        )

    # 1. 断言输出张量维度
    assert c_probs.shape == (1, 2)
    assert b_probs.shape == (1, 1)
    assert s_dist.shape == (1, 5)
    assert s_exp.shape == (1, 1)

    # 2. 断言概率约束
    assert torch.isclose(c_probs.sum(), torch.tensor(1.0, device=device), atol=1e-3)
    assert torch.isclose(s_dist.sum(), torch.tensor(1.0, device=device), atol=1e-3)
    assert (b_probs >= 0.0).all() and (b_probs <= 1.0).all()
    assert (s_exp >= 1.0).all() and (s_exp <= 5.0).all()

