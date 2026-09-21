import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSetAttentionHead(nn.Module):
    """
    单位超球面余弦归一化集合注意力决策头 (Choice 动态多选一原语)
    """
    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 256,
        temperature: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(
        self,
        query_hidden: torch.Tensor,
        candidates_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.q_proj.weight.dtype
        q = self.q_proj(query_hidden.to(target_dtype))
        if q.dim() == 2:
            q = q.unsqueeze(1)
        
        k = self.k_proj(candidates_hidden.to(target_dtype))

        q_norm = F.normalize(q, p=2, dim=-1, eps=self.eps)
        k_norm = F.normalize(k, p=2, dim=-1, eps=self.eps)

        raw_scores = torch.bmm(q_norm, k_norm.transpose(1, 2)).squeeze(1)
        scaled_logits = raw_scores / self.temperature

        if mask is not None:
            scaled_logits = scaled_logits.masked_fill(~mask, -1e4)

        probabilities = F.softmax(scaled_logits, dim=-1)
        return scaled_logits, probabilities


class BooleanScalarHead(nn.Module):
    """
    是非判定标量决策头 (Boolean 原语)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, query_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.classifier.weight.dtype
        logits = self.classifier(query_hidden.to(target_dtype))
        probabilities = torch.sigmoid(logits)
        return logits, probabilities


class ScoreOrdinalHead(nn.Module):
    """
    序数等级期望打分决策头 (Score 原语)
    """
    def __init__(self, hidden_size: int, max_levels: int = 5):
        super().__init__()
        self.max_levels = max_levels
        self.classifier = nn.Linear(hidden_size, max_levels)
        self.register_buffer("level_weights", torch.arange(1.0, max_levels + 1.0, dtype=torch.float32))

    def forward(
        self,
        query_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = self.classifier.weight.dtype
        logits = self.classifier(query_hidden.to(target_dtype))
        
        if active_levels is not None and active_levels < self.max_levels:
            mask = torch.arange(self.max_levels, device=logits.device) >= active_levels
            logits = logits.masked_fill(mask, -1e4)

        distribution = F.softmax(logits, dim=-1)
        expected_score = torch.sum(distribution * self.level_weights.to(distribution.dtype), dim=-1, keepdim=True)
        return logits, distribution, expected_score


class UnifiedDecisionHeads(nn.Module):
    """
    双正交多态决策总成 (Unified Decision Assembly)
    """
    def __init__(
        self,
        hidden_size: int,
        choice_proj_dim: int = 256,
        temperature: float = 0.1,
        score_max_levels: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.choice_head = CosineSetAttentionHead(hidden_size, choice_proj_dim, temperature)
        self.boolean_head = BooleanScalarHead(hidden_size)

        self.impact_head = ScoreOrdinalHead(hidden_size, score_max_levels)
        self.sensitivity_head = ScoreOrdinalHead(hidden_size, score_max_levels)

        self.score_head = self.impact_head

    def forward_choice(
        self,
        q_hidden: torch.Tensor,
        c_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.choice_head(q_hidden, c_hidden, mask)

    def forward_boolean(self, q_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.boolean_head(q_hidden)

    def forward_impact(
        self,
        q_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.impact_head(q_hidden, active_levels)

    def forward_sensitivity(
        self,
        q_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sensitivity_head(q_hidden, active_levels)

    def forward_score(
        self,
        q_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.impact_head(q_hidden, active_levels)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """兼容旧版单头模型权重平滑过渡：若检测到旧单头权重，自动安全分发至双头"""
        old_w_key = prefix + "score_head.classifier.weight"
        old_b_key = prefix + "score_head.classifier.bias"
        old_l_key = prefix + "score_head.level_weights"

        imp_w_key = prefix + "impact_head.classifier.weight"
        imp_b_key = prefix + "impact_head.classifier.bias"
        imp_l_key = prefix + "impact_head.level_weights"

        sens_w_key = prefix + "sensitivity_head.classifier.weight"
        sens_b_key = prefix + "sensitivity_head.classifier.bias"
        sens_l_key = prefix + "sensitivity_head.level_weights"

        # 映射权重与偏置
        if old_w_key in state_dict and imp_w_key not in state_dict:
            state_dict[imp_w_key] = state_dict[old_w_key].clone()
        if old_b_key in state_dict and imp_b_key not in state_dict:
            state_dict[imp_b_key] = state_dict[old_b_key].clone()

        if old_w_key in state_dict and sens_w_key not in state_dict:
            state_dict[sens_w_key] = state_dict[old_w_key].clone()
        if old_b_key in state_dict and sens_b_key not in state_dict:
            state_dict[sens_b_key] = state_dict[old_b_key].clone()

        # 映射等级常数缓冲区 (解决 Missing key level_weights)
        if old_l_key in state_dict and imp_l_key not in state_dict:
            state_dict[imp_l_key] = state_dict[old_l_key].clone()
        elif imp_l_key not in state_dict and hasattr(self, "impact_head"):
            state_dict[imp_l_key] = self.impact_head.level_weights.clone()

        if old_l_key in state_dict and sens_l_key not in state_dict:
            state_dict[sens_l_key] = state_dict[old_l_key].clone()
        elif sens_l_key not in state_dict and hasattr(self, "sensitivity_head"):
            state_dict[sens_l_key] = self.sensitivity_head.level_weights.clone()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )
