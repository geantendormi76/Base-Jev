import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from contracts import (
    BatchDecisionRequest,
    CandidateOption,
    ChoiceQuestion,
    DecisionHeadsSpec,
    DecisionPrimitive,
    ProductionRecipe,
    ScoreQuestion,
    SynthesizedSample,
    TypedStateSnapshot,
)
from joint_rlcd_loss import JointCalibratedRLCDLoss
from sequence_packer import SequencePacker
from system_one_model import SystemOneModel


class FormattedDecisionDataset(Dataset):
    def __init__(self, data_path: Union[str, Path]):
        self.samples: List[SynthesizedSample] = []
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"数据集文件不存在: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_data = json.loads(line)
                    sample = SynthesizedSample.model_validate(raw_data)
                    self.samples.append(sample)
                except Exception as err:
                    print(f"[!] 跳过第 {line_idx} 行未通过强类型契约校验的数据: {err}", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SynthesizedSample:
        return self.samples[idx]


class FactoryTrainer:
    def __init__(
        self,
        recipe: ProductionRecipe,
        model: SystemOneModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.recipe = recipe
        self.model = model
        self.tokenizer = tokenizer
        self.packer = SequencePacker(tokenizer)

        weights = recipe.training.loss_weights
        self.loss_fn = JointCalibratedRLCDLoss(
            weight_soft_ce=weights.soft_ce,
            weight_brier=weights.brier,
            weight_entropy=weights.entropy,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.attach_lora(recipe.training)

        # 核心解冻：显式确保双正交打分头的分类器矩阵参与反向传播，彻底打破 3.20 天花板！
        for param in self.model.heads.impact_head.parameters():
            param.requires_grad = True
        for param in self.model.heads.sensitivity_head.parameters():
            param.requires_grad = True

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"  [✓] 成功挂载并激活可训练参数: {trainable_count:,} 个 (LoRA + 独立物理双头)")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=recipe.training.learning_rate,
            weight_decay=0.01,
        )

    def sample_to_request_and_target(
        self, 
        sample: SynthesizedSample
    ) -> Tuple[BatchDecisionRequest, Dict[str, Any]]:
        state = TypedStateSnapshot(
            module=sample.framework,
            ast_node=sample.ast_node_type,
            scope=sample.path,
            attributes=[
                ("host", sample.host),
                ("method", sample.method.upper()),
                ("params", ",".join(sample.extracted_params)),
                ("code", sample.code_slice),
            ],
        )

        choices = []
        scores = []

        target_meta: Dict[str, Any] = {
            "primitive": sample.primitive,
            "is_ambiguous": sample.is_ambiguous,
            "sample_id": sample.sample_id,
        }

        if sample.primitive == DecisionPrimitive.CHOICE:
            choices.append(
                ChoiceQuestion(
                    id=sample.sample_id,
                    query=sample.query,
                    candidates=sample.candidates,
                )
            )
            target_meta["target_distribution"] = sample.target_distribution or [0.5, 0.5]

        elif sample.primitive == DecisionPrimitive.DUAL_SCORE:
            if sample.target_scores and "impact" in sample.target_scores and "sensitivity" in sample.target_scores:
                imp_t = sample.target_scores["impact"]
                sens_t = sample.target_scores["sensitivity"]

                scores.append(ScoreQuestion(
                    id=f"{sample.sample_id}_impact",
                    query=f"评估接口状态变更破坏力: {sample.method} {sample.path[:60]}",
                    active_levels=5
                ))
                scores.append(ScoreQuestion(
                    id=f"{sample.sample_id}_sensitivity",
                    query=f"评估接口资产敏感度: {sample.method} {sample.path[:60]}",
                    active_levels=5
                ))

                target_meta["target_scores"] = {
                    "impact": {
                        "target_distribution": imp_t.distribution,
                        "target_expected": imp_t.expected_score,
                    },
                    "sensitivity": {
                        "target_distribution": sens_t.distribution,
                        "target_expected": sens_t.expected_score,
                    }
                }
            else:
                raise ValueError(f"样本 [{sample.sample_id}] 缺少完整 target_scores！")

        elif sample.primitive == DecisionPrimitive.SCORE:
            dist = sample.target_distribution or [0.2]*5
            exp_val = sample.target_expected if sample.target_expected is not None else 3.0
            scores.append(ScoreQuestion(id=sample.sample_id, query=sample.query, active_levels=len(dist)))
            target_meta["target_distribution"] = dist
            target_meta["target_expected"] = exp_val

        request = BatchDecisionRequest(
            state=state,
            choices=choices,
            booleans=[],
            scores=scores,
        )
        return request, target_meta

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        step_count = 0
        grad_accum_steps = self.recipe.training.gradient_accumulation_steps

        for batch_idx, sample in enumerate(dataloader):
            request, target = self.sample_to_request_and_target(sample)
            packed = self.packer.pack_request(request)

            input_ids = packed.input_ids.to(self.device)
            attention_mask = packed.attention_mask.to(self.device)

            hidden_states = self.model.extract_hidden_states(input_ids, attention_mask)
            seq_hidden = hidden_states.squeeze(0)

            sample_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            is_ambig_tensor = torch.tensor([target["is_ambiguous"]], device=self.device, dtype=torch.float32)

            if target["primitive"] == DecisionPrimitive.CHOICE and packed.choice_slices:
                c_slice = packed.choice_slices[0]
                q_vec = seq_hidden[c_slice.query_token_idx : c_slice.query_token_idx + 1]
                c_vecs = seq_hidden[c_slice.candidate_token_indices].unsqueeze(0)

                _, probs = self.model.heads.forward_choice(q_vec, c_vecs)
                target_dist = torch.tensor(
                    [target["target_distribution"]], device=self.device, dtype=torch.float32
                )
                sample_loss += self.loss_fn.compute_choice_loss(probs, target_dist, is_ambig_tensor)

            elif target["primitive"] == DecisionPrimitive.DUAL_SCORE and len(packed.score_slices) >= 2:
                # 破坏力 -> impact_head
                s_imp = packed.score_slices[0]
                q_imp_vec = seq_hidden[s_imp.query_token_idx : s_imp.query_token_idx + 1]
                _, imp_dist, imp_exp = self.model.heads.forward_impact(q_imp_vec)
                imp_tdist = torch.tensor([target["target_scores"]["impact"]["target_distribution"]], device=self.device, dtype=torch.float32)
                imp_texp = torch.tensor([[target["target_scores"]["impact"]["target_expected"]]], device=self.device, dtype=torch.float32)
                loss_imp = self.loss_fn.compute_score_loss(imp_dist, imp_exp, imp_tdist, imp_texp, is_ambig_tensor)

                # 敏感度 -> sensitivity_head
                s_sens = packed.score_slices[1]
                q_sens_vec = seq_hidden[s_sens.query_token_idx : s_sens.query_token_idx + 1]
                _, sens_dist, sens_exp = self.model.heads.forward_sensitivity(q_sens_vec)
                sens_tdist = torch.tensor([target["target_scores"]["sensitivity"]["target_distribution"]], device=self.device, dtype=torch.float32)
                sens_texp = torch.tensor([[target["target_scores"]["sensitivity"]["target_expected"]]], device=self.device, dtype=torch.float32)
                loss_sens = self.loss_fn.compute_score_loss(sens_dist, sens_exp, sens_tdist, sens_texp, is_ambig_tensor)

                sample_loss += (loss_imp + loss_sens)

            scaled_loss = sample_loss / grad_accum_steps
            scaled_loss.backward()

            total_loss += sample_loss.item()
            step_count += 1

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.recipe.training.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = total_loss / max(1, step_count)
        return avg_loss

    def save_checkpoint(self, output_dir: Union[str, Path]) -> None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        lora_dir = out_path / "backbone_lora"
        self.model.backbone.save_pretrained(str(lora_dir))

        heads_path = out_path / "decision_heads.pt"
        torch.save(self.model.heads.state_dict(), str(heads_path))

        self.tokenizer.save_pretrained(str(out_path))
