import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
import torch
from joint_rlcd_loss import JointCalibratedRLCDLoss


def test_joint_choice_loss_clean_vs_ambiguous():
    """验证 RLCD 损失：纯净样本收敛，模糊样本触发最大熵抑制"""
    loss_fn = JointCalibratedRLCDLoss()

    # 1. 纯净独热样本 (确凿证据)
    probs_clean = torch.tensor([[0.95, 0.05]], requires_grad=True)
    target_clean = torch.tensor([[1.0, 0.0]])
    is_ambiguous_clean = torch.tensor([0.0])

    loss_clean = loss_fn.compute_choice_loss(probs_clean, target_clean, is_ambiguous_clean)
    loss_clean.backward()

    assert not torch.isnan(loss_clean)
    assert probs_clean.grad is not None
    assert torch.all(torch.isfinite(probs_clean.grad))

    # 2. 对抗模糊样本 (软标签五五开)
    probs_ambig = torch.tensor([[0.51, 0.49]], requires_grad=True)
    target_ambig = torch.tensor([[0.5, 0.5]])
    is_ambiguous_ambig = torch.tensor([1.0])

    loss_ambig = loss_fn.compute_choice_loss(probs_ambig, target_ambig, is_ambiguous_ambig)
    loss_ambig.backward()

    assert not torch.isnan(loss_ambig)
    assert probs_ambig.grad is not None


def test_joint_boolean_and_score_loss_numerical_stability():
    """验证 Boolean 与 Score 极端概率下的数值稳定性 (杜绝 NaN)"""
    loss_fn = JointCalibratedRLCDLoss()

    # 极端下溢测试 (0.0 与 1.0 边界)
    b_probs = torch.tensor([[0.0], [1.0]], requires_grad=True)
    b_targets = torch.tensor([[0.0], [1.0]])
    b_ambig = torch.tensor([0.0, 1.0])

    b_loss = loss_fn.compute_boolean_loss(b_probs, b_targets, b_ambig)
    b_loss.backward()

    assert not torch.isnan(b_loss)
    assert not torch.isnan(b_probs.grad).any()

    # Score 序数分布与连续分值
    s_dist = torch.tensor([[0.01, 0.02, 0.05, 0.12, 0.80]], requires_grad=True)
    s_exp = torch.tensor([[4.68]], requires_grad=True)
    target_dist = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    target_exp = torch.tensor([[5.0]])
    s_ambig = torch.tensor([0.0])

    s_loss = loss_fn.compute_score_loss(s_dist, s_exp, target_dist, target_exp, s_ambig)
    s_loss.backward()

    assert not torch.isnan(s_loss)
    assert not torch.isnan(s_dist.grad).any()

