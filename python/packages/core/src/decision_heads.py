from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSetAttentionHead(nn.Module):
    """单位超球面余弦归一化集合注意力决策头 (Choice 动态多选一原语)"""
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
    """是非判定标量决策头 (Boolean 原语)"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, query_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.classifier.weight.dtype
        logits = self.classifier(query_hidden.to(target_dtype))
        probabilities = torch.sigmoid(logits)
        return logits, probabilities

class ScoreOrdinalHead(nn.Module):
    """序数等级期望打分决策头 (Score 原语)"""
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
    通用工业级多态多正交决策总成 (纯泛化实现)
    支持: Choice, Boolean 与任意 N 维正交序数打分头
    """
    def __init__(
        self,
        hidden_size: int,
        choice_proj_dim: int = 256,
        temperature: float = 0.1,
        score_dimensions: Optional[List[str]] = None,
        score_max_levels: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.choice_head = CosineSetAttentionHead(hidden_size, choice_proj_dim, temperature)
        self.boolean_head = BooleanScalarHead(hidden_size)

        # 动态注册 N 维正交打分头
        dims = score_dimensions if score_dimensions else ["impact", "sensitivity"]
        self.score_heads = nn.ModuleDict({
            dim_name: ScoreOrdinalHead(hidden_size, score_max_levels)
            for dim_name in dims
        })

    # 向下兼容历史别名访问 (使用 __contains__ 与 __getitem__，彻底根绝 ModuleDict.get() 异常)
    @property
    def impact_head(self) -> ScoreOrdinalHead:
        if "impact" in self.score_heads:
            return self.score_heads["impact"]
        return next(iter(self.score_heads.values()))

    @property
    def sensitivity_head(self) -> ScoreOrdinalHead:
        if "sensitivity" in self.score_heads:
            return self.score_heads["sensitivity"]
        return list(self.score_heads.values())[-1]

    @property
    def score_head(self) -> ScoreOrdinalHead:
        return self.impact_head

    def forward_choice(
        self,
        q_hidden: torch.Tensor,
        c_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.choice_head(q_hidden, c_hidden, mask)

    def forward_boolean(self, q_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.boolean_head(q_hidden)

    def forward_score_dim(
        self,
        dim_name: str,
        q_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dim_name not in self.score_heads:
            raise KeyError(f"未注册的评分维度: '{dim_name}'. 可用维度: {list(self.score_heads.keys())}")
        return self.score_heads[dim_name](q_hidden, active_levels)

    def forward_impact(
        self,
        q_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_score_dim("impact", q_hidden, active_levels)

    def forward_sensitivity(
        self,
        q_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_score_dim("sensitivity", q_hidden, active_levels)

    def forward_score(
        self,
        q_hidden: torch.Tensor,
        active_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_impact(q_hidden, active_levels)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        三代权重自愈平滑迁移器:
        1. 自动将旧版 impact_head / sensitivity_head 映射入 score_heads
        2. 自动吸收历史 score_head.* 别名残留，消除 unexpected_keys 报错
        3. 确保 strict=True 严格模式下 100% 绿通
        """
        # 1. 迁移 v2 双正交头命名
        for old_dim in ["impact", "sensitivity"]:
            old_pfx = f"{prefix}{old_dim}_head."
            new_pfx = f"{prefix}score_heads.{old_dim}."
            matched_keys = [k for k in list(state_dict.keys()) if k.startswith(old_pfx)]
            for ok in matched_keys:
                nk = ok.replace(old_pfx, new_pfx)
                if nk not in state_dict and old_dim in self.score_heads:
                    state_dict[nk] = state_dict[ok]
                state_dict.pop(ok, None)

        # 2. 迁移并吸收 v1 单头或别名残留 (score_head.*)
        old_score_pfx = f"{prefix}score_head."
        legacy_score_keys = [k for k in list(state_dict.keys()) if k.startswith(old_score_pfx)]
        for ok in legacy_score_keys:
            nk = ok.replace(old_score_pfx, f"{prefix}score_heads.impact.")
            if nk not in state_dict and "impact" in self.score_heads:
                state_dict[nk] = state_dict[ok]
            state_dict.pop(ok, None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
