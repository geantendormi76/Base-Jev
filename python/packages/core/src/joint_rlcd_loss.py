import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointCalibratedRLCDLoss(nn.Module):
    """
    联合认识论校准损失函数 (Joint Calibrated RLCD Loss)
    
    数学准则:
    1. 软目标交叉熵 (Soft-CE) + Brier 均方概率校准 (Brier Score)
    2. 冲突模糊样本最大熵正则化 (Maximum Entropy Penalty)
    3. 全程在 FP32 进行数值运算，防止 log(p) 下溢出出现 NaN
    """
    def __init__(
        self,
        weight_soft_ce: float = 1.0,
        weight_brier: float = 0.5,
        weight_entropy: float = 0.25,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.weight_soft_ce = weight_soft_ce
        self.weight_brier = weight_brier
        self.weight_entropy = weight_entropy
        self.eps = eps

    def compute_choice_loss(
        self,
        probabilities: torch.Tensor,
        target_distribution: torch.Tensor,
        is_ambiguous: torch.Tensor,
    ) -> torch.Tensor:
        """
        probabilities: (batch_size, K)
        target_distribution: (batch_size, K) 软概率标签向量
        is_ambiguous: (batch_size,) 1.0 为对抗/模糊样本，0.0 为纯净确定样本
        """
        p = torch.clamp(probabilities.float(), min=self.eps, max=1.0)
        q = target_distribution.float()
        k_dim = p.shape[-1]

        # 1. 软交叉熵: - sum(q * log(p))
        ce_loss = -torch.sum(q * torch.log(p), dim=-1)

        # 2. Brier 均方概率校准: 1/(2K) * sum((p - q)^2)
        brier_loss = (1.0 / (2.0 * k_dim)) * torch.sum((p - q) ** 2, dim=-1)

        # 3. 最大熵正则化: 对模糊样本惩罚过低的信息熵，迫使其输出一碗水端平的平坦概率
        # Shannon Entropy H(p) = - sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p), dim=-1)
        entropy_penalty = -is_ambiguous.float() * self.weight_entropy * entropy

        total_sample_loss = (
            self.weight_soft_ce * ce_loss
            + self.weight_brier * brier_loss
            + entropy_penalty
        )
        return torch.mean(total_sample_loss)

    def compute_boolean_loss(
        self,
        probabilities: torch.Tensor,
        target_labels: torch.Tensor,
        is_ambiguous: torch.Tensor,
    ) -> torch.Tensor:
        """
        probabilities: (batch_size, 1)
        target_labels: (batch_size, 1) float 连续概率 [0.0, 1.0]
        """
        p = torch.clamp(probabilities.float(), min=self.eps, max=1.0 - self.eps)
        y = target_labels.float()

        # 1. 软二元交叉熵
        bce_loss = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).squeeze(-1)

        # 2. Brier 校准
        brier_loss = 0.5 * ((p - y) ** 2).squeeze(-1)

        # 3. 二元香农熵
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).squeeze(-1)
        entropy_penalty = -is_ambiguous.float() * self.weight_entropy * entropy

        total_sample_loss = (
            self.weight_soft_ce * bce_loss
            + self.weight_brier * brier_loss
            + entropy_penalty
        )
        return torch.mean(total_sample_loss)

    def compute_score_loss(
        self,
        distribution: torch.Tensor,
        expected_score: torch.Tensor,
        target_distribution: torch.Tensor,
        target_expected: torch.Tensor,
        is_ambiguous: torch.Tensor,
    ) -> torch.Tensor:
        """
        distribution: (batch_size, M)
        expected_score: (batch_size, 1)
        """
        p = torch.clamp(distribution.float(), min=self.eps, max=1.0)
        q = target_distribution.float()

        # 1. 序数分类交叉熵
        ce_loss = -torch.sum(q * torch.log(p), dim=-1)

        # 2. 连续期望分值回归 MSE 误差
        score_mse = 0.5 * ((expected_score.squeeze(-1) - target_expected.squeeze(-1)) ** 2)

        # 3. 熵正则化
        entropy = -torch.sum(p * torch.log(p), dim=-1)
        entropy_penalty = -is_ambiguous.float() * self.weight_entropy * entropy

        total_sample_loss = (
            self.weight_soft_ce * ce_loss
            + self.weight_brier * score_mse
            + entropy_penalty
        )
        return torch.mean(total_sample_loss)
