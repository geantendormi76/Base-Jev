import sys
from pathlib import Path

# 保证无论从根目录还是子目录调用，src 均在搜索路径中
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import pytest
from contracts import (
    BatchDecisionRequest,
    BatchDecisionResponse,
    BooleanQuestion,
    CandidateOption,
    ChoiceOutcome,
    ChoiceQuestion,
    ModelPassport,
    PassportArtifacts,
    PassportBenchmarks,
    PassportMetrics,
    PrecisionTier,
    ProductionRecipe,
    ScoreOutcome,
    ScoreQuestion,
    TypedStateSnapshot,
)


def test_production_recipe_defaults_and_validation():
    recipe = ProductionRecipe(
        recipe_id="recipe-test-01",
        product_name="Test Scorer",
        domain="security",
        datasets={"train_path": "data/train.jsonl", "eval_path": "data/eval.jsonl"},
    )
    assert recipe.backbone.hidden_size == 1024
    assert recipe.heads.temperature == 0.1
    assert recipe.export.precision == PrecisionTier.FP16
    assert recipe.training.loss_weights.entropy == 0.25


def test_model_passport_serialization():
    passport = ModelPassport(
        model_uuid="uuid-12345",
        product_name="Test Passport",
        recipe_id="recipe-test-01",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        file_size_bytes=1312000000,
        metrics=PassportMetrics(
            clean_accuracy=1.0,
            ambiguous_confidence_suppression=0.49,
            brier_score=0.035,
            expected_calibration_error=0.02,
        ),
        benchmarks=PassportBenchmarks(
            device_name="NVIDIA RTX 3060",
            runtime="DirectML",
            steady_latency_ms=37.92,
            p99_latency_ms=41.2,
            vram_allocated_mb=1251.6,
        ),
        artifacts=PassportArtifacts(
            onnx_relative_path="models/system_one_unified.onnx",
            tokenizer_relative_path="models/tokenizer.json",
        ),
    )
    data = json.loads(passport.model_dump_json())
    assert data["model_uuid"] == "uuid-12345"
    assert data["metrics"]["clean_accuracy"] == 1.0
    assert data["benchmarks"]["steady_latency_ms"] == 37.92


def test_decision_request_response_symmetry():
    req = BatchDecisionRequest(
        state=TypedStateSnapshot(
            module="auth",
            ast_node="CallExpression",
            scope="handle_login",
            attributes=[("method", "POST"), ("path", "/api/login")],
        ),
        choices=[
            ChoiceQuestion(
                id="q_action",
                query="Choose next action",
                candidates=[CandidateOption(index=0, text="allow"), CandidateOption(index=1, text="block")],
            )
        ],
        booleans=[BooleanQuestion(id="q_risk", query="Is this high risk?")],
        scores=[ScoreQuestion(id="q_score", query="Threat score", active_levels=5)],
    )
    req_json = req.model_dump_json()
    assert "handle_login" in req_json

    resp = BatchDecisionResponse(
        choice_outcomes={
            "q_action": ChoiceOutcome(selected_index=1, confidence=0.98, probabilities=[0.02, 0.98])
        },
        boolean_outcomes={"q_risk": 0.95},
        score_outcomes={
            "q_score": ScoreOutcome(distribution=[0.01, 0.02, 0.05, 0.12, 0.80], expected_score=4.68)
        },
        latency_ms=37.8,
    )
    resp_json = resp.model_dump_json()
    assert "q_action" in resp_json
