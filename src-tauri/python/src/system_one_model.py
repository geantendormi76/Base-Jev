from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from peft import LoraConfig, get_peft_model

from contracts import DecisionHeadsSpec, TrainingHyperparameters
from decision_heads import UnifiedDecisionHeads


def extract_hidden_size_from_backbone(backbone: PreTrainedModel) -> int:
    """
    纯泛化隐层维度萃取器 (支持纯文本基模、嵌套多模态基模与物理层兜底)
    """
    config = getattr(backbone, "config", None)
    if config is not None:
        # 1. 传统纯文本架构顶级属性 (Qwen2, LLaMA, Mistral, Gemma)
        if hasattr(config, "hidden_size") and getattr(config, "hidden_size") is not None:
            return int(config.hidden_size)

        # 2. 复合多模态架构子配置 (Qwen3.5, MiniCPM-V)
        for sub_attr in ["text_config", "llm_config", "language_config"]:
            sub_cfg = getattr(config, sub_attr, None)
            if sub_cfg is not None and hasattr(sub_cfg, "hidden_size"):
                val = getattr(sub_cfg, "hidden_size")
                if val is not None:
                    return int(val)

    # 3. 物理张量层直读兜底 (直接从 RMSNorm 或 Embedding 提取客观几何维度)
    for module in backbone.modules():
        # 从最后归一化层读取 (如 Qwen3_5RMSNorm(2048))
        if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            if "norm" in module.__class__.__name__.lower() and module.weight.ndim == 1:
                return int(module.weight.shape[0])

    raise AttributeError("无法从当前基模结构或配置中确定隐层特征维度 hidden_size")


class SystemOneModel(nn.Module):
    """
    工业级 System 1 强类型纯前向决策模型
    """
    def __init__(
        self,
        backbone: PreTrainedModel,
        heads_spec: Optional[DecisionHeadsSpec] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config

        # 纯泛化提取隐层维度
        self.hidden_size = extract_hidden_size_from_backbone(backbone)

        spec = heads_spec or DecisionHeadsSpec()
        self.heads = UnifiedDecisionHeads(
            hidden_size=self.hidden_size,
            choice_proj_dim=spec.choice_projection_dim,
            temperature=spec.temperature,
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
        """
        从本地路径或开源模型 ID 加载纯表征骨干 (零 lm_head 负载)
        """
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
        """
        为骨干注入表征学习 LoRA
        """
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
        """
        执行纯前向传播，输出文本 Token 的末层隐层特征
        返回: (batch_size, seq_len, hidden_size)
        """
        device = self.backbone.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 兼容纯文本输入模式的前向调用
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # 兼容不同模型架构的输出命名 (last_hidden_state 或 text_model_output)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            raise AttributeError("未能从前向传播输出中解析末层隐层特征")
