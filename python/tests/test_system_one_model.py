import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-1.7B-Base"

import pytest
import torch
from system_one_model import SystemOneModel


def test_load_local_qwen3_backbone_and_decouple_lm_head():
    """验证物理资产：加载下载完成的 Qwen3-1.7B-Base 权重并断言 lm_head 已彻底剔除"""
    if not MODEL_DIR.exists():
        pytest.skip(f"本地 {MODEL_DIR} 资产未就绪，跳过物理权重加载测试")

    print(f"\n[+] 正在挂载本地物理纯文本基模: {MODEL_DIR.resolve()}")
    model = SystemOneModel.from_pretrained_base(
        model_path_or_id=MODEL_DIR,
    )

    # 1. 验证隐层维度自适应萃取 (断言 hidden_size 正确提取且大于 0)
    assert model.hidden_size > 0
    assert model.heads.hidden_size == model.hidden_size
    print(f"[✓] 成功动态萃取 Qwen3 隐层特征维度: {model.hidden_size}")

    # 2. 验证解耦定理：骨干模型绝不携带 lm_head 属性 (节省超过 1GB 显存)
    assert not hasattr(model.backbone, "lm_head")

    # 3. 验证表征前向传播 (构造单步 8 Token 输入)
    dummy_input = torch.tensor([[101, 2056, 3891, 4012, 55, 66, 77, 88]], dtype=torch.long)
    with torch.no_grad():
        hidden_states = model.extract_hidden_states(dummy_input)

    assert hidden_states.shape == (1, 8, model.hidden_size)

    # 4. 验证决策头对接 (模拟 Query 取自末位 Token，候选取自前 3 个 Token)
    q_vec = hidden_states[:, -1, :]
    c_vecs = hidden_states[:, :3, :]

    _, choice_probs = model.heads.forward_choice(q_vec, c_vecs)
    _, bool_prob = model.heads.forward_boolean(q_vec)
    _, score_dist, score_exp = model.heads.forward_score(q_vec)

    assert choice_probs.shape == (1, 3)
    assert bool_prob.shape == (1, 1)
    assert score_dist.shape == (1, 5)
    assert score_exp.shape == (1, 1)
    assert torch.isclose(choice_probs.sum(), torch.tensor(1.0, device=choice_probs.device), atol=1e-5)

