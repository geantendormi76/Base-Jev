from pathlib import Path
import sys
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-0.6B-Base"
if not MODEL_DIR.exists():
    MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-1.7B-Base"

from onnx_export_wrapper import SystemOneOnnxWrapper
from system_one_model import SystemOneModel


def test_onnx_wrapper_forward_and_gather_correctness():
    """验证 ONNX 包装器：Gather 算子正确提取 Query 与 Candidate 特征并完成真·双正交头输出"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SystemOneModel.from_pretrained_base(
        MODEL_DIR,
        device_map=None,
        torch_dtype=torch.float32,
    )
    wrapper = SystemOneOnnxWrapper(model.backbone, model.heads)

    # 构造动态输入: 序列长度 16，2 个候选，1 个布尔，1 个破坏力，1 个敏感度
    seq_len = 16
    input_ids = torch.randint(100, 1000, (1, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    choice_q_pos = torch.tensor([15], dtype=torch.long, device=device)
    choice_cand_pos = torch.tensor([[5, 10]], dtype=torch.long, device=device)
    choice_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=device)
    bool_q_pos = torch.tensor([15], dtype=torch.long, device=device)
    impact_q_pos = torch.tensor([15], dtype=torch.long, device=device)
    sensitivity_q_pos = torch.tensor([15], dtype=torch.long, device=device)

    (
        choice_probs,
        bool_probs,
        impact_dist,
        impact_expected,
        sensitivity_dist,
        sensitivity_expected,
    ) = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        choice_q_pos=choice_q_pos,
        choice_cand_pos=choice_cand_pos,
        choice_mask=choice_mask,
        bool_q_pos=bool_q_pos,
        impact_q_pos=impact_q_pos,
        sensitivity_q_pos=sensitivity_q_pos,
    )

    # 1. 断言输出张量维度
    assert choice_probs.shape == (1, 2)
    assert bool_probs.shape == (1, 1)
    assert impact_dist.shape == (1, 5)
    assert impact_expected.shape == (1, 1)
    assert sensitivity_dist.shape == (1, 5)
    assert sensitivity_expected.shape == (1, 1)

    # 2. 断言概率约束
    assert torch.allclose(choice_probs.sum(dim=-1), torch.tensor([1.0], device=device), atol=1e-4)
    assert torch.allclose(impact_dist.sum(dim=-1), torch.tensor([1.0], device=device), atol=1e-4)
    assert torch.allclose(sensitivity_dist.sum(dim=-1), torch.tensor([1.0], device=device), atol=1e-4)
    assert 1.0 <= impact_expected.item() <= 5.0
    assert 1.0 <= sensitivity_expected.item() <= 5.0
