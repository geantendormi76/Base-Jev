from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import PreTrainedTokenizerBase

from contracts import (
    BatchDecisionRequest,
    BooleanQuestion,
    ChoiceQuestion,
    ScoreQuestion,
    TypedStateSnapshot,
)

CAND_ANCHOR_TOKEN = "<|cand_anchor|>"
QUERY_ANCHOR_TOKEN = "<|query_anchor|>"


@dataclass
class ChoiceAnchorSlice:
    question_id: str
    query_token_idx: int
    candidate_token_indices: List[int]


@dataclass
class ScalarAnchorSlice:
    question_id: str
    query_token_idx: int


@dataclass
class PackedSequenceBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    choice_slices: List[ChoiceAnchorSlice] = field(default_factory=list)
    boolean_slices: List[ScalarAnchorSlice] = field(default_factory=list)
    score_slices: List[ScalarAnchorSlice] = field(default_factory=list)
    raw_prompt: str = ""


class SequencePacker:
    """
    单序列多问题因果打包器 (Sequence Packer)
    
    因果偏序铁律:
    State Context ≺ Candidates ≺ Queries
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self._ensure_special_anchors()

    def _ensure_special_anchors(self) -> None:
        """确保原子锚点标记已注册，防止被 BPE 切碎为多子词"""
        special_tokens_to_add = []
        for token in [CAND_ANCHOR_TOKEN, QUERY_ANCHOR_TOKEN]:
            if token not in self.tokenizer.get_vocab():
                special_tokens_to_add.append(token)

        if special_tokens_to_add:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens_to_add}
            )

        self.cand_anchor_id = self.tokenizer.convert_tokens_to_ids(CAND_ANCHOR_TOKEN)
        self.query_anchor_id = self.tokenizer.convert_tokens_to_ids(QUERY_ANCHOR_TOKEN)

    def pack_request(self, request: BatchDecisionRequest) -> PackedSequenceBatch:
        """
        将 BatchDecisionRequest 按照因果偏序严格装配为单一 Token 序列并追踪所有锚点索引
        """
        # 1. 装配状态上下文 (State Context)
        state_lines = [
            "[STATE_CONTEXT]",
            f"Module: {request.state.module}",
            f"AST_Node: {request.state.ast_node}",
            f"Scope: {request.state.scope}",
        ]
        for k, v in request.state.attributes:
            state_lines.append(f"{k}: {v}")
        state_text = "\n".join(state_lines) + "\n\n"

        prompt = state_text

        # 2. 依次装配 Choice 问题及其候选 (Candidates precede Query)
        choice_slices: List[ChoiceAnchorSlice] = []
        for choice in request.choices:
            prompt += f"[CHOICE_QUESTION: {choice.id}]\n"
            cand_indices = []
            for opt in choice.candidates:
                prompt += f"{CAND_ANCHOR_TOKEN} Option_{opt.index}: {opt.text}\n"
            prompt += f"{QUERY_ANCHOR_TOKEN} Query: {choice.query}\n\n"

        # 3. 依次装配 Boolean 问题
        boolean_slices: List[ScalarAnchorSlice] = []
        for b in request.booleans:
            prompt += f"[BOOLEAN_QUESTION: {b.id}]\n"
            prompt += f"{QUERY_ANCHOR_TOKEN} Query: {b.query}\n\n"

        # 4. 依次装配 Score 序数问题
        score_slices: List[ScalarAnchorSlice] = []
        for s in request.scores:
            prompt += f"[SCORE_QUESTION: {s.id}]\n"
            prompt += f"{QUERY_ANCHOR_TOKEN} Query: {s.query} (MaxLevels: {s.active_levels})\n\n"

        # 5. 执行 Tokenization
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded.input_ids[0]  # (seq_len,)
        attention_mask = encoded.attention_mask[0]

        # 6. 精确定位锚点 Token 在输入张量中的下标
        all_cand_positions = (input_ids == self.cand_anchor_id).nonzero(as_tuple=True)[0].tolist()
        all_query_positions = (input_ids == self.query_anchor_id).nonzero(as_tuple=True)[0].tolist()

        query_cursor = 0
        cand_cursor = 0

        # 分配 Choice 锚点
        for choice in request.choices:
            num_cands = len(choice.candidates)
            c_pos = all_cand_positions[cand_cursor : cand_cursor + num_cands]
            q_pos = all_query_positions[query_cursor]
            choice_slices.append(
                ChoiceAnchorSlice(
                    question_id=choice.id,
                    query_token_idx=q_pos,
                    candidate_token_indices=c_pos,
                )
            )
            cand_cursor += num_cands
            query_cursor += 1

        # 分配 Boolean 锚点
        for b in request.booleans:
            q_pos = all_query_positions[query_cursor]
            boolean_slices.append(
                ScalarAnchorSlice(
                    question_id=b.id,
                    query_token_idx=q_pos,
                )
            )
            query_cursor += 1

        # 分配 Score 锚点
        for s in request.scores:
            q_pos = all_query_positions[query_cursor]
            score_slices.append(
                ScalarAnchorSlice(
                    question_id=s.id,
                    query_token_idx=q_pos,
                )
            )
            query_cursor += 1

        return PackedSequenceBatch(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            choice_slices=choice_slices,
            boolean_slices=boolean_slices,
            score_slices=score_slices,
            raw_prompt=prompt,
        )
