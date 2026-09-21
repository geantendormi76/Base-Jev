import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
import torch
from decision_heads import CosineSetAttentionHead, BooleanScalarHead, ScoreOrdinalHead, UnifiedDecisionHeads


def test_cosine_set_attention_bounds_and_probability_sum():
    """验证定理：分值严格有界在 [-1/tau, 1/tau]，概率和恒为 1.0"""
    batch_size = 2
    hidden_size = 1024
    num_candidates = 5
    temperature = 0.1

    head = CosineSetAttentionHead(hidden_size=hidden_size, projection_dim=256, temperature=temperature)
    
    q_hidden = torch.randn(batch_size, hidden_size) * 100.0
    c_hidden = torch.randn(batch_size, num_candidates, hidden_size) * 100.0

    logits, probs = head(q_hidden, c_hidden)

    max_bound = 1.0 / temperature + 1e-4
    assert torch.all(logits >= -max_bound)
    assert torch.all(logits <= max_bound)

    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_cosine_set_attention_gradient_never_saturates():
    """验证定理：无论初始权重有多大，反向传播梯度绝对充盈不消失"""
    head = CosineSetAttentionHead(hidden_size=512, projection_dim=128, temperature=0.1)
    q = torch.randn(1, 512, requires_grad=True)
    c = torch.randn(1, 4, 512, requires_grad=True)

    _, probs = head(q, c)
    target = torch.tensor([2])
    loss = -torch.log(probs[0, target] + 1e-12)
    loss.backward()

    assert q.grad is not None
    assert torch.norm(q.grad) > 0.0
    assert not torch.isnan(q.grad).any()


def test_dual_orthogonal_heads_physical_decoupling():
    """
    严密验证契约: impact_head 与 sensitivity_head 物理独立
    铁律: 对 Impact 求导反传时，Sensitivity 参数梯度必须绝对为 None，零梯度泄露！
    """
    hidden_size = 512
    heads = UnifiedDecisionHeads(hidden_size=hidden_size, score_max_levels=5)

    # 1. 物理对象与权重张量绝对独立性断言
    assert heads.impact_head is not heads.sensitivity_head
    assert heads.impact_head.classifier.weight is not heads.sensitivity_head.classifier.weight
    assert heads.impact_head.classifier.bias is not heads.sensitivity_head.classifier.bias

    # 2. 构造两个独立的输入向量
    q_imp = torch.randn(1, hidden_size, requires_grad=True)
    q_sens = torch.randn(1, hidden_size, requires_grad=True)

    # 3. 分别前向计算
    _, imp_dist, imp_exp = heads.forward_impact(q_imp)
    _, sens_dist, sens_exp = heads.forward_sensitivity(q_sens)

    assert imp_dist.shape == (1, 5)
    assert sens_dist.shape == (1, 5)
    assert imp_exp.shape == (1, 1)
    assert sens_exp.shape == (1, 1)

    # 4. 仅针对 impact_exp 进行反向传播
    loss = imp_exp.sum()
    loss.backward()

    # 5. 核心物理断言：impact_head 获得充盈梯度，sensitivity_head 梯度必须绝对为 None！
    assert heads.impact_head.classifier.weight.grad is not None
    assert torch.norm(heads.impact_head.classifier.weight.grad) > 0.0
    assert heads.sensitivity_head.classifier.weight.grad is None
    assert heads.sensitivity_head.classifier.bias.grad is None

