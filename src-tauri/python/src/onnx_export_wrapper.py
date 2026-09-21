from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from decision_heads import UnifiedDecisionHeads


class SystemOneOnnxWrapper(nn.Module):
    """
    System 1 纯前向静态计算图 ONNX 导出包装器
    
    设计准则:
    1. 骨干 Transformer 与超球打分头融合成单一前向图。
    2. 使用标准 torch.index_select (ONNX Gather) 提取锚点特征，消灭非标切片。
    3. 全流程支持动态序列长度与动态候选分支伸缩。
    """
    def __init__(self, backbone: nn.Module, heads: UnifiedDecisionHeads):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(
        self,
        input_ids: torch.Tensor,               # (1, seq_len)
        attention_mask: torch.Tensor,          # (1, seq_len)
        choice_q_pos: torch.Tensor,            # (num_choices,)
        choice_cand_pos: torch.Tensor,         # (num_choices, max_cands)
        choice_mask: torch.Tensor,             # (num_choices, max_cands) float32/bool
        bool_q_pos: torch.Tensor,              # (num_booleans,)
        score_q_pos: torch.Tensor,             # (num_scores,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            choice_probs:   (num_choices, max_cands)
            bool_probs:     (num_booleans, 1)
            score_dist:     (num_scores, max_levels)
            score_expected: (num_scores, 1)
        """
        # 1. 骨干前向传播，提取末层隐层特征
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
        seq_hidden = last_hidden.squeeze(0)      # (seq_len, hidden_size)

        # 2. Choice 原语特征汇聚 (利用 Gather 算子)
        num_choices = choice_q_pos.shape[0]
        max_cands = choice_cand_pos.shape[1]

        # 提取 Query 向量: (num_choices, hidden_size)
        choice_q_vecs = torch.index_select(seq_hidden, 0, choice_q_pos)

        # 展平候选索引执行批量 Gather: (num_choices * max_cands, hidden_size)
        flat_cand_indices = choice_cand_pos.reshape(-1)
        flat_cand_vecs = torch.index_select(seq_hidden, 0, flat_cand_indices)
        choice_c_vecs = flat_cand_vecs.reshape(num_choices, max_cands, -1)

        # 超球注意力打分
        _, choice_probs = self.heads.forward_choice(
            choice_q_vecs,
            choice_c_vecs,
            mask=choice_mask.bool(),
        )

        # 3. Boolean 原语特征汇聚
        bool_q_vecs = torch.index_select(seq_hidden, 0, bool_q_pos)
        _, bool_probs = self.heads.forward_boolean(bool_q_vecs)

        # 4. Score 原语特征汇聚
        score_q_vecs = torch.index_select(seq_hidden, 0, score_q_pos)
        _, score_dist, score_expected = self.heads.forward_score(score_q_vecs)

        return choice_probs, bool_probs, score_dist, score_expected
