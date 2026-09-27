from pathlib import Path
import sys
import torch
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_heads import CosineSetAttentionHead, UnifiedDecisionHeads

def test_cosine_set_attention_bounds_and_probability_sum():
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
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)

def test_cosine_set_attention_gradient_never_saturates():
    head = CosineSetAttentionHead(hidden_size=512, projection_dim=128, temperature=0.1)
    q = torch.randn(1, 512, requires_grad=True)
    c = torch.randn(1, 4, 512, requires_grad=True)

    logits, probs = head(q, c)
    target = torch.tensor([2])
    loss = -torch.log(probs[0, target] + 1e-12)
    loss.backward()

    assert q.grad is not None
    assert c.grad is not None
    assert torch.norm(q.grad) > 0.0
    assert torch.norm(c.grad) > 0.0

def test_dual_orthogonal_heads_physical_decoupling():
    """严密验证契约: impact_head 与 sensitivity_head 物理独立"""
    hidden_size = 512
    heads = UnifiedDecisionHeads(hidden_size=hidden_size, score_max_levels=5)

    assert heads.impact_head is not heads.sensitivity_head
    assert heads.impact_head.classifier.weight is not heads.sensitivity_head.classifier.weight

    q_imp = torch.randn(1, hidden_size, requires_grad=True)
    q_sens = torch.randn(1, hidden_size, requires_grad=True)

    _, _, imp_exp = heads.forward_impact(q_imp)
    _, _, sens_exp = heads.forward_sensitivity(q_sens)

    loss = imp_exp.sum()
    loss.backward()

    assert any(p.grad is not None for p in heads.impact_head.parameters())
    assert all(p.grad is None for p in heads.sensitivity_head.parameters())

def test_multi_orthogonal_heads_dynamic_registration_and_gradient_isolation():
    """验证新契约: 动态注册 4 维多正交 (残值 x 暗病 x 流动性 x 假卡真伪)，验证梯度完全隔离"""
    hidden_size = 512
    dims = ["impact", "sensitivity", "liquidity", "authenticity"]
    heads = UnifiedDecisionHeads(hidden_size=hidden_size, score_dimensions=dims, score_max_levels=5)

    assert len(heads.score_heads) == 4
    for d in dims:
        assert d in heads.score_heads

    # 对 liquidity 求导，其余 3 头零渗透
    q_liq = torch.randn(1, hidden_size, requires_grad=True)
    _, _, liq_exp = heads.forward_score_dim("liquidity", q_liq)
    liq_exp.sum().backward()

    assert any(p.grad is not None for p in heads.score_heads["liquidity"].parameters())
    for other in ["impact", "sensitivity", "authenticity"]:
        assert all(p.grad is None for p in heads.score_heads[other].parameters())
