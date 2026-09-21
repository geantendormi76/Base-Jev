from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class PrecisionTier(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"


class DecisionPrimitive(str, Enum):
    CHOICE = "Choice"
    BOOLEAN = "Boolean"
    SCORE = "Score"
    DUAL_SCORE = "DualScore"


class ProvenanceType(str, Enum):
    REAL = "REAL"            # 物理真实锚点
    SYNTHETIC = "SYNTHETIC"  # 程序化/跨框架合成


class SampleCategory(str, Enum):
    POSITIVE = "POSITIVE"                            # 正向业务
    INVALID_NOISE = "INVALID_NOISE"                  # 拒识噪音 (OOD)
    ADVERSARIAL_AMBIGUOUS = "ADVERSARIAL_AMBIGUOUS"  # 认识论安全熔断样本


class ScoreDimensionTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: List[float] = Field(description="5档归一化后验概率分布 (和恒为1.0)")
    expected_score: float = Field(description="精确代数加权期望值 sum(m*p)")


class BackboneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(default="Qwen/Qwen3-0.6B-Base", description="开源基模 ID 或本地路径")
    hidden_size: int = Field(default=1024, ge=128, description="Transformer 骨干隐层特征维度")
    max_seq_len: int = Field(default=2048, ge=256, description="单序列最大允许 Token 长度")


class DecisionHeadsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice_enabled: bool = Field(default=True, description="是否启用 Choice 多选一决策头")
    choice_projection_dim: int = Field(default=256, ge=64, description="超球注意力投影子空间维度 d")
    temperature: float = Field(default=0.1, gt=0.0, description="余弦注意力温度系数 tau")
    boolean_enabled: bool = Field(default=True, description="是否启用 Boolean 标量判断头")
    score_enabled: bool = Field(default=True, description="是否启用 Score 序数期望打分头")
    score_max_levels: int = Field(default=5, ge=2, le=20, description="Score 序数最大评级档位")


class LossWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soft_ce: float = Field(default=1.0, ge=0.0, description="软标签交叉熵权重")
    brier: float = Field(default=0.5, ge=0.0, description="Brier 分数均方误差权重")
    entropy: float = Field(default=0.25, ge=0.0, description="对抗模糊样本最大熵惩罚权重")


class TrainingHyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=2e-4, gt=0.0)
    epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    ambiguity_anchor_ratio: float = Field(default=0.15, ge=0.05, le=0.5)


class DatasetPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str = Field(description="训练集 JSONL 相对或绝对路径")
    eval_path: str = Field(description="测试/盲测集 JSONL 相对或绝对路径")


class ExportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: PrecisionTier = Field(default=PrecisionTier.FP16)
    output_filename: str = Field(default="system_one_unified.onnx")


class ProductionRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_id: str = Field(description="配方唯一 ID")
    product_name: str = Field(description="产出产品业务名称")
    domain: str = Field(description="业务领域分类")
    backbone: BackboneSpec = Field(default_factory=BackboneSpec)
    heads: DecisionHeadsSpec = Field(default_factory=DecisionHeadsSpec)
    training: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    datasets: DatasetPaths
    export: ExportSpec = Field(default_factory=ExportSpec)


# ==============================================================================
# 上游蒸馏标准样本强契约 (SynthesizedSample)
# ==============================================================================

class CandidateOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    text: str = Field(min_length=1)


class SynthesizedSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    provenance: ProvenanceType
    category: SampleCategory
    primitive: DecisionPrimitive
    framework: str
    host: str
    method: str
    path: str
    extracted_params: List[str] = Field(default_factory=list)
    code_slice: str
    query: str
    candidates: List[CandidateOption] = Field(default_factory=list)
    is_ambiguous: float = Field(default=0.0, ge=0.0, le=1.0)
    ast_node_type: str = Field(default="CallExpression")

    # 双正交打分结构 (Impact 破坏力 + Sensitivity 敏感度)
    target_scores: Optional[Dict[str, ScoreDimensionTarget]] = Field(
        default=None, description="双正交目标得分字典: impact 与 sensitivity"
    )

    # 兼容单维度旧字段别名
    target_distribution: Optional[List[float]] = None
    target_expected: Optional[float] = None
    target_label: Optional[float] = None


# ==============================================================================
# 运行时决策多原语请求与响应契约
# ==============================================================================

class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    candidates: List[CandidateOption] = Field(min_length=2, max_length=255)


class BooleanQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    active_levels: int = Field(default=5, ge=2, le=20)


class TypedStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    ast_node: str
    scope: str
    attributes: List[Tuple[str, str]] = Field(default_factory=list)


class ChoiceOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_index: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: List[float]


class ScoreOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: List[float]
    expected_score: float


class BatchDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: TypedStateSnapshot
    choices: List[ChoiceQuestion] = Field(default_factory=list)
    booleans: List[BooleanQuestion] = Field(default_factory=list)
    scores: List[ScoreQuestion] = Field(default_factory=list)


class BatchDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice_outcomes: Dict[str, ChoiceOutcome] = Field(default_factory=dict)
    boolean_outcomes: Dict[str, float] = Field(default_factory=dict)
    score_outcomes: Dict[str, ScoreOutcome] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0.0)


# ==============================================================================
# 模型出厂护照契约 (Model Passport)
# ==============================================================================

class PassportMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clean_accuracy: float = Field(ge=0.0, le=1.0)
    ambiguous_confidence_suppression: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0)
    expected_calibration_error: float = Field(ge=0.0, le=1.0)


class PassportBenchmarks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_name: str
    runtime: str
    steady_latency_ms: float = Field(gt=0.0)
    p99_latency_ms: float = Field(gt=0.0)
    vram_allocated_mb: float = Field(ge=0.0)
    supported_candidate_range: Tuple[int, int] = Field(default=(2, 255))


class PassportArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    onnx_relative_path: str
    tokenizer_relative_path: str


class ModelPassport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passport_version: str = Field(default="v1.0.0")
    model_uuid: str = Field(description="产出模型的全局唯一 UUID")
    product_name: str
    recipe_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sha256: str = Field(description="ONNX 单文件的 SHA-256 哈希值")
    file_size_bytes: int = Field(gt=0)
    primitives: List[str] = Field(default_factory=lambda: ["Choice", "Boolean", "Score", "DualScore"])
    metrics: PassportMetrics
    benchmarks: PassportBenchmarks
    artifacts: PassportArtifacts
