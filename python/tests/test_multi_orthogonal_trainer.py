from pathlib import Path
import sys
import torch
import pytest
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contracts import (
    ProductionRecipe,
    DecisionHeadsSpec,
    SynthesizedSample,
    ScoreDimensionTarget,
    DecisionPrimitive,
    ProvenanceType,
    SampleCategory,
)
from system_one_model import SystemOneModel
from trainer import FactoryTrainer

def test_trainer_multi_orthogonal_3_dimensions_loop(tmp_path):
    """验证训练引擎: 动态支持 3 维正交损失反传与参数解冻 (残值 x 暗病 x 流动性)"""
    dims = ["impact", "sensitivity", "liquidity"]
    sample = SynthesizedSample(
        sample_id="SYNTH-MULTI-001",
        provenance=ProvenanceType.SYNTHETIC,
        category=SampleCategory.POSITIVE,
        primitive=DecisionPrimitive.MULTI_SCORE,
        framework="TestGPU",
        host="test.com",
        method="SELL",
        path="/test",
        code_slice="标的:3090|价格:5600",
        query="三维综合评测",
        target_scores={
            "impact": ScoreDimensionTarget(distribution=[0.0, 0.1, 0.2, 0.5, 0.2], expected_score=3.8),
            "sensitivity": ScoreDimensionTarget(distribution=[0.8, 0.2, 0.0, 0.0, 0.0], expected_score=1.2),
            "liquidity": ScoreDimensionTarget(distribution=[0.1, 0.1, 0.6, 0.1, 0.1], expected_score=3.0),
        }
    )

    recipe = ProductionRecipe(
        recipe_id="test-multi-01",
        product_name="Test Multi Product",
        domain="hardware",
        heads=DecisionHeadsSpec(score_dimensions=dims),
        datasets={"train_path": "fake", "eval_path": "fake"}
    )

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(128, 128)
            self.v_proj = torch.nn.Linear(128, 128)
            self.config = type("Config", (), {"hidden_size": 128})()

        @property
        def device(self):
            return self.q_proj.weight.device

        def forward(self, input_ids, attention_mask=None, return_dict=True):
            batch, seq = input_ids.shape
            last_hidden = torch.randn(batch, seq, 128, device=input_ids.device, requires_grad=True)
            return type("Output", (), {"last_hidden_state": last_hidden})()

        def save_pretrained(self, path):
            pass

    dummy_backbone = DummyBackbone()
    model = SystemOneModel(backbone=dummy_backbone, heads_spec=recipe.heads)

    model_dir = Path(__file__).resolve().parents[2] / "models" / "Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    trainer = FactoryTrainer(recipe=recipe, model=model, tokenizer=tokenizer)

    # 1. 验证动态挂载与全量解冻
    assert len(trainer.model.heads.score_heads) == 3
    for d in dims:
        assert d in trainer.model.heads.score_heads

    # 2. 验证 sample_to_request_and_target 映射闭环
    req, meta = trainer.sample_to_request_and_target(sample)
    assert len(req.scores) == 3
    assert meta["dimension_order"] == dims
    assert "liquidity" in meta["target_scores"]

    # 3. 记录优化前权重快照
    w_before = {d: trainer.model.heads.score_heads[d].classifier.weight.clone() for d in dims}

    # 4. 执行单步训练推演与反向传播
    test_loader = DataLoader([sample], batch_size=1, collate_fn=lambda b: b[0])
    loss = trainer.train_epoch(test_loader, epoch=1)
    assert loss > 0.0

    # 5. 核心客观断言：断言全部 3 个头均接收到了独立梯度并在 optimizer.step() 后发生真实的参数位移！
    for d in dims:
        w_after = trainer.model.heads.score_heads[d].classifier.weight
        weight_delta = torch.norm(w_after - w_before[d]).item()
        assert weight_delta > 0.0, f"维度 {d} 的参数矩阵未发生物理更新 (更新量: {weight_delta})"
