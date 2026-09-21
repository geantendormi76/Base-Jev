use base_jev_core::*;
use std::collections::HashMap;

#[test]
fn test_production_recipe_serde_roundtrip() {
    let recipe = ProductionRecipe {
        recipe_id: "recipe-rust-01".to_string(),
        product_name: "Rust Test Product".to_string(),
        domain: "network-security".to_string(),
        backbone: BackboneSpec::default(),
        heads: DecisionHeadsSpec::default(),
        training: TrainingHyperparameters::default(),
        datasets: DatasetPaths {
            train_path: "data/train.jsonl".to_string(),
            eval_path: "data/eval.jsonl".to_string(),
        },
        export: ExportSpec::default(),
    };

    let json_str = serde_json::to_string_pretty(&recipe).unwrap();
    let restored: ProductionRecipe = serde_json::from_str(&json_str).unwrap();

    assert_eq!(restored.recipe_id, "recipe-rust-01");
    assert_eq!(restored.backbone.hidden_size, 1024);
    assert_eq!(restored.export.precision, PrecisionTier::FP16);
}

#[test]
fn test_model_passport_serde_roundtrip() {
    let passport = ModelPassport {
        passport_version: "v1.0.0".to_string(),
        model_uuid: "uuid-rust-999".to_string(),
        product_name: "Invar Surface Scorer".to_string(),
        recipe_id: "recipe-invar-01".to_string(),
        created_at: "2026-09-19T00:00:00Z".to_string(),
        sha256: "aabbcc112233".to_string(),
        file_size_bytes: 1251620000,
        primitives: vec!["Choice".to_string(), "Score".to_string()],
        metrics: PassportMetrics {
            clean_accuracy: 1.0,
            ambiguous_confidence_suppression: 0.485,
            brier_score: 0.041,
            expected_calibration_error: 0.015,
        },
        benchmarks: PassportBenchmarks {
            device_name: "RTX 3060 DirectML".to_string(),
            runtime: "DirectML".to_string(),
            steady_latency_ms: 37.92,
            p99_latency_ms: 40.5,
            vram_allocated_mb: 1251.62,
            supported_candidate_range: (2, 255),
        },
        artifacts: PassportArtifacts {
            onnx_relative_path: "models/system_one_unified.onnx".to_string(),
            tokenizer_relative_path: "models/tokenizer.json".to_string(),
        },
    };

    let json_str = serde_json::to_string(&passport).unwrap();
    let restored: ModelPassport = serde_json::from_str(&json_str).unwrap();

    assert_eq!(restored.model_uuid, "uuid-rust-999");
    assert_eq!(restored.benchmarks.steady_latency_ms, 37.92);
}

#[test]
fn test_decision_batch_serde_roundtrip() {
    let req = BatchDecisionRequest {
        state: TypedStateSnapshot {
            module: "router".to_string(),
            ast_node: "CallExpression".to_string(),
            scope: "handle_packet".to_string(),
            attributes: vec![("port".to_string(), "8080".to_string())],
        },
        choices: vec![ChoiceQuestion {
            id: "q1".to_string(),
            query: "Select route".to_string(),
            candidates: vec![
                CandidateOption { index: 0, text: "eth0".to_string() },
                CandidateOption { index: 1, text: "eth1".to_string() },
            ],
        }],
        booleans: vec![BooleanQuestion {
            id: "b1".to_string(),
            query: "Drop packet?".to_string(),
        }],
        scores: vec![ScoreQuestion {
            id: "s1".to_string(),
            query: "Danger level".to_string(),
            active_levels: 5,
        }],
    };

    let json_str = serde_json::to_string(&req).unwrap();
    let restored: BatchDecisionRequest = serde_json::from_str(&json_str).unwrap();
    assert_eq!(restored.state.module, "router");
    assert_eq!(restored.choices[0].candidates.len(), 2);

    let mut choice_outcomes = HashMap::new();
    choice_outcomes.insert(
        "q1".to_string(),
        ChoiceOutcome {
            selected_index: 0,
            confidence: 0.97,
            probabilities: vec![0.97, 0.03],
        },
    );

    let resp = BatchDecisionResponse {
        choice_outcomes,
        boolean_outcomes: HashMap::new(),
        score_outcomes: HashMap::new(),
        latency_ms: 36.5,
    };

    let resp_json = serde_json::to_string(&resp).unwrap();
    let restored_resp: BatchDecisionResponse = serde_json::from_str(&resp_json).unwrap();
    assert_eq!(restored_resp.choice_outcomes.get("q1").unwrap().selected_index, 0);
}
