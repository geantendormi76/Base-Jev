from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from peft import LoraConfig, get_peft_model
from contracts import DecisionHeadsSpec, TrainingHyperparameters
from decision_heads import UnifiedDecisionHeads

def extract_hidden_size_from_backbone(backbone: PreTrainedModel) -> int:
    config = getattr(backbone, "config", None)
    if config is not None:
        if hasattr(config, "hidden_size") and getattr(config, "hidden_size") is not None:
            return int(config.hidden_size)
        for sub_attr in ["text_config", "llm_config", "language_config"]:
            sub_cfg = getattr(config, sub_attr, None)
            if sub_cfg is not None and hasattr(sub_cfg, "hidden_size"):
                val = getattr(sub_cfg, "hidden_size")
                if val is not None:
                    return int(val)
    for module in backbone.modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            if "norm" in module.__class__.__name__.lower() and module.weight.ndim == 1:
                return int(module.weight.shape[0])
    raise AttributeError("无法从当前基模结构或配置中确定隐层特征维度 hidden_size")

class SystemOneModel(nn.Module):
    """工业级 System 1 强类型纯前向决策模型"""
    def __init__(
        self,
        backbone: PreTrainedModel,
        heads_spec: Optional[DecisionHeadsSpec] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.hidden_size = extract_hidden_size_from_backbone(backbone)
        spec = heads_spec or DecisionHeadsSpec()
        
        # 核心注入：动态支持多正交打分维度列表
        dims = getattr(spec, "score_dimensions", ["impact", "sensitivity"])
        self.heads = UnifiedDecisionHeads(
            hidden_size=self.hidden_size,
            choice_proj_dim=spec.choice_projection_dim,
            temperature=spec.temperature,
            score_dimensions=dims,
            score_max_levels=spec.score_max_levels,
        )

    @classmethod
    def from_pretrained_base(
        cls,
        model_path_or_id: Union[str, Path],
        heads_spec: Optional[DecisionHeadsSpec] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "SystemOneModel":
        path_str = str(model_path_or_id)
        config = AutoConfig.from_pretrained(path_str, trust_remote_code=True)
        eff_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        eff_device_map = device_map or ("auto" if torch.cuda.is_available() else None)
        backbone = AutoModel.from_pretrained(
            path_str,
            config=config,
            torch_dtype=eff_dtype,
            device_map=eff_device_map,
            trust_remote_code=True,
        )
        model = cls(backbone=backbone, heads_spec=heads_spec)
        if torch.cuda.is_available():
            model.heads = model.heads.to(backbone.device)
        return model

    def attach_lora(self, training_spec: Optional[TrainingHyperparameters] = None) -> None:
        spec = training_spec or TrainingHyperparameters()
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj", "out_proj",
        ]
        lora_config = LoraConfig(
            r=spec.lora_r,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=None,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = self.backbone.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            raise AttributeError("未能从前向传播输出中解析末层隐层特征")
