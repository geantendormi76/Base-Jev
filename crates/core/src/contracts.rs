use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionTier {
    FP16,
    INT8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackboneSpec {
    pub model_id: String,
    pub hidden_size: usize,
    pub max_seq_len: usize,
}

impl Default for BackboneSpec {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen3-0.6B-Base".to_string(),
            hidden_size: 1024,
            max_seq_len: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionHeadsSpec {
    pub choice_enabled: bool,
    pub choice_projection_dim: usize,
    pub temperature: f32,
    pub boolean_enabled: bool,
    pub score_enabled: bool,
    pub score_max_levels: usize,
}

impl Default for DecisionHeadsSpec {
    fn default() -> Self {
        Self {
            choice_enabled: true,
            choice_projection_dim: 256,
            temperature: 0.1,
            boolean_enabled: true,
            score_enabled: true,
            score_max_levels: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LossWeights {
    pub soft_ce: f32,
    pub brier: f32,
    pub entropy: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            soft_ce: 1.0,
            brier: 0.5,
            entropy: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub lora_r: usize,
    pub lora_alpha: usize,
    pub lora_dropout: f32,
    pub max_grad_norm: f32,
    pub loss_weights: LossWeights,
    pub ambiguity_anchor_ratio: f32,
}

impl Default for TrainingHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            epochs: 4,
            batch_size: 4,
            gradient_accumulation_steps: 4,
            lora_r: 16,
            lora_alpha: 32,
            lora_dropout: 0.05,
            max_grad_norm: 1.0,
            loss_weights: LossWeights::default(),
            ambiguity_anchor_ratio: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetPaths {
    pub train_path: String,
    pub eval_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExportSpec {
    pub precision: PrecisionTier,
    pub output_filename: String,
}

impl Default for ExportSpec {
    fn default() -> Self {
        Self {
            precision: PrecisionTier::FP16,
            output_filename: "system_one_unified.onnx".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProductionRecipe {
    pub recipe_id: String,
    pub product_name: String,
    pub domain: String,
    #[serde(default)]
    pub backbone: BackboneSpec,
    #[serde(default)]
    pub heads: DecisionHeadsSpec,
    #[serde(default)]
    pub training: TrainingHyperparameters,
    pub datasets: DatasetPaths,
    #[serde(default)]
    pub export: ExportSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PassportMetrics {
    pub clean_accuracy: f32,
    pub ambiguous_confidence_suppression: f32,
    pub brier_score: f32,
    pub expected_calibration_error: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PassportBenchmarks {
    pub device_name: String,
    pub runtime: String,
    pub steady_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub vram_allocated_mb: f32,
    #[serde(default = "default_candidate_range")]
    pub supported_candidate_range: (usize, usize),
}

fn default_candidate_range() -> (usize, usize) {
    (2, 255)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PassportArtifacts {
    pub onnx_relative_path: String,
    pub tokenizer_relative_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPassport {
    #[serde(default = "default_passport_version")]
    pub passport_version: String,
    pub model_uuid: String,
    pub product_name: String,
    pub recipe_id: String,
    pub created_at: String,
    pub sha256: String,
    pub file_size_bytes: u64,
    #[serde(default = "default_primitives")]
    pub primitives: Vec<String>,
    pub metrics: PassportMetrics,
    pub benchmarks: PassportBenchmarks,
    pub artifacts: PassportArtifacts,
}

fn default_passport_version() -> String {
    "v1.0.0".to_string()
}

fn default_primitives() -> Vec<String> {
    vec!["Choice".to_string(), "Boolean".to_string(), "Score".to_string()]
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateOption {
    pub index: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChoiceQuestion {
    pub id: String,
    pub query: String,
    pub candidates: Vec<CandidateOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BooleanQuestion {
    pub id: String,
    pub query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScoreQuestion {
    pub id: String,
    pub query: String,
    #[serde(default = "default_score_levels")]
    pub active_levels: u32,
}

fn default_score_levels() -> u32 {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TypedStateSnapshot {
    pub module: String,
    pub ast_node: String,
    pub scope: String,
    #[serde(default)]
    pub attributes: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChoiceOutcome {
    pub selected_index: u32,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreOutcome {
    pub distribution: Vec<f32>,
    pub expected_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchDecisionRequest {
    pub state: TypedStateSnapshot,
    #[serde(default)]
    pub choices: Vec<ChoiceQuestion>,
    #[serde(default)]
    pub booleans: Vec<BooleanQuestion>,
    #[serde(default)]
    pub scores: Vec<ScoreQuestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchDecisionResponse {
    #[serde(default)]
    pub choice_outcomes: HashMap<String, ChoiceOutcome>,
    #[serde(default)]
    pub boolean_outcomes: HashMap<String, f32>,
    #[serde(default)]
    pub score_outcomes: HashMap<String, ScoreOutcome>,
    pub latency_ms: f32,
}
