# AI Development Handoff Specification

> **Document Type:** Cross-session engineering state snapshot & recovery contract
> **Project:** base-jev (System-1 Cognitive Reflex Engine in Dual-Engine System)
> **Canonical Root:** `C:\dev\Base-Jev` & `C:\dev\Invar` & `C:\dev\distiller`
> **Snapshot Date:** 2026-09-22
> **Document Version:** 7.0.0 (Standardized Monorepo Edition & Zero-Contamination Verified)
> **Primary Purpose:** A new AI session must be able to reconstruct the exact engineering and research state from this document plus current repository source, tests, and configuration without relying on conversational history.

---

## 0. Handoff Metadata

| Item | Value | Evidence Level | Notes |
|---|---|---|---|
| Project Identity | base-jev + Invar | [FACT] | System-1 Fast Sensor (23ms) + System-2 Deep Verification |
| Repository Architecture | AI Engineering Monorepo | [FACT] | Fully aligned to official Mother Scaffold (crates / python / configs / data / models) |
| Model Factory Root | `C:\dev\Base-Jev` | [FACT] | Discriminative SFT Factory & DirectML TensorCore Runtime |
| Distillation Workshop | `C:\dev\distiller` | [FACT] | Dual-Teacher Distillation (Qwen-27B Local + Gemini API) |
| Downstream Verification | `C:\dev\Invar` | [FACT] | DualTrack Comparator, IDOR Differential Tester, Gold Set Evaluator |
| Current Backbone | `models/Qwen3-0.6B-Base` | [FACT] | 1024D Hidden Size, `lm_head` physically stripped |
| Production ONNX SHA256 | `908db8140ab25a32...` | [FACT] | 1251.11 MB single-file FP16 static graph |
| Gold Set Ground Truth | 60 cards, 10 true high-risk | [FACT] | Zero-contamination verified (`Train ∩ Gold = ∅`) |
| **Model Recall on Gold Set** | **80.00% (TP=8/10, FN=2)** | [FACT] | Precision: 36.36%, F1: 0.5000 (Outperformed Rule 50% / 0.3448) |
| **Pool B Discrepancy Recall** | **100.00% (TP=5/5, FN=0)** | [FACT] | Precision: 62.50%, F1: 0.7692 (Rule completely failed: 0%) |
| Rust Core Test Status | **3/3 PASSED (100%)** | [FACT] | `cargo test --workspace` verified |
| Python Core Test Status | **37/37 PASSED (100%)** | [FACT] | `cd python; uv run pytest` verified (4 standard skips) |
| Ghost Paths Audit | **0 Remaining** | [FACT] | Full audit passed via S5 Path Audit |

---

## 1. Project Identity & Triad Architecture

[DECISION]
The system is an industrial Triad Architecture with clear boundaries:
1. **distiller** (`C:\dev\distiller`): Expert knowledge distillation workshop. Dual-teacher distillation (Local Qwen-27B + Cloud Gemini Flash) with Tree-sitter AST extraction and physical Gold Firewall.
2. **base-jev** (`C:\dev\Base-Jev` - This Repo): System-1 cognitive reflex frontend. Pure discriminative SFT (Soft-CE + Brier Score + Entropy loss). DirectML FP16 TensorCore execution (~23.08ms steady latency). Pure tensor classification, zero generative token overhead.
3. **Invar** (`C:\dev\Invar`): System-2 execution and deep verification. DualTrack Comparator, IDOR differential testing, bounded symbolic execution, and future Cost-Aware GRPO.

---

## 2. Standardized Monorepo Layout (Post S1~S8 Migration)

```text
C:/dev/Base-Jev/
├── apps/
│   └── desktop/                  # React + TypeScript + Tailwind + Tauri v2 product shell
├── crates/
│   └── core/                     # Rust DirectML TensorCore runtime engine & base-jev-qa tool
├── python/
│   ├── packages/
│   │   └── core/                 # System-1 decision heads, sequence packer, joint loss
│   ├── scripts/                  # train_factory_model.py, export_to_onnx.py, generate_triage_predictions.py
│   ├── tests/                    # 37 Python test suite (contracts, heads, loss, packer)
│   └── pyproject.toml            # uv-managed workspace configuration
├── configs/
│   ├── base/                     # committed defaults (project.toml)
│   ├── profiles/                 # recipe_invar_0.6b.json, research.toml
│   ├── registry/                 # backends.toml (declarative registry)
│   └── schemas/                  # config.schema.json
├── data/
│   ├── inbox/, raw/, interim/, processed/, fixtures/, manifests/
├── models/
│   ├── Qwen3-0.6B-Base/          # Pretrained base weights (1024D)
│   ├── Qwen3-0.6B-Base-SystemOne-Invar/  # LoRA adapter & decision heads checkpoint
│   ├── system_one_unified.onnx   # DirectML unified static graph (1.25 GB)
│   └── registry/                 # Model passports and metadata
├── artifacts/
│   ├── exports/, reports/, metrics/, packages/
├── scripts/
│   ├── audit/                    # 18 forensic audit & lineage scripts
│   ├── data/                     # download_dataset.py, download_model.py, preprocess_fuzzybench_jev.py
│   └── build/                    # consolidate_onnx.py
├── docs/
│   ├── ai/                       # HANDOFF.md, REPOSITORY_CLOSURE.md, FILE_DECISIONS.md, PATH_MIGRATION.md
│   ├── architecture/             # System architecture & runtime boundaries
│   ├── decisions/                # ADRs
│   └── research/                 # Experimental evidence & benchmark logs
├── tools/
│   ├── bootstrap/                # ai_monorepo_scaffold.py
│   └── repomix/                  # 1_.json (official context packaging config)
├── tmp/                          # Disposable ephemeral workspace (repomix/ etc.)
├── Cargo.toml                    # Root Rust workspace (manages crates/core, apps/*/src-tauri)
├── package.json                  # Root Node/pnpm workspace
├── pnpm-workspace.yaml
├── .editorconfig
├── .gitattributes
├── .gitignore                    # Whitelists !.gitkeep, ignores target, .venv, runs, tmp
├── README.md                     # Monorepo directory map & runbook
└── HANDOFF.md                    # Root handoff contract (mirrored from docs/ai/HANDOFF.md)
```

---

## 3. Verified Test & Run Commands

### Rust Core Workspace
```powershell
cargo test --workspace
```
* **Status**: `ok. 3 passed; 0 failed; 0 ignored; finished in 0.00s`
* **Test cases**: `test_model_passport_serde_roundtrip`, `test_production_recipe_serde_roundtrip`, `test_decision_batch_serde_roundtrip`.

### Python Core Workspace (uv)
```powershell
cd python
uv run pytest
cd ..
```
* **Status**: `37 passed, 4 skipped in 3.69s` (4 skipped are GPU/full weight dependencies).
* **Test suites**: `test_contracts`, `test_decision_heads`, `test_endpoint_validation`, `test_fuzzybench_preprocessing`, `test_joint_rlcd_loss`, `test_onnx_export`, `test_sequence_packer`, `test_system_one_model`.

---

## 4. Architectural Invariants (铁律不变量)

* **I-01: Zero Contamination**: `TRAIN ∩ GOLD = ∅`. Gold cards (60 endpoints) must NEVER leak into training datasets. Physical firewall verified by E3 audit.
* **I-02: 1:1 Prediction Parity**: `len(predictions) == len(endpoints)` (1221/1221 aligned, zero silent omissions).
* **I-03: Tensor-Only Output**: `base-jev` outputs probabilities and ordinal expectation tensors; zero autoregressive text generation overhead.
* **I-04: Fail-Closed Joins**: Evaluator and comparator joins must fail-fast on unmatched surface IDs (zero silent 3.0 fallbacks).
* **I-05: Causal Packing Order**: In sequence packing: `State Context` ≺ `Candidates` ≺ `Queries`.
* **I-06: Dual Orthogonal Heads**: Safety risk is strictly factorized into **Impact** (state change / destruction) and **Sensitivity** (asset confidentiality).

---

## 5. Active Data & Model Contracts

* **Production Recipe**: `configs/profiles/recipe_invar_0.6b.json`
* **Base Model**: `models/Qwen3-0.6B-Base` (1024 hidden size)
* **Checkpoint**: `models/Qwen3-0.6B-Base-SystemOne-Invar/` (`backbone_lora/` + `decision_heads.pt`)
* **Unified ONNX**: `models/system_one_unified.onnx` (DirectML FP16)
* **Active Distillation Datasets**:
  - `data/API_distilled/distilled_invar_train.jsonl` (400 train, Gemini Flash)
  - `data/API_distilled/distilled_invar_eval.jsonl` (100 eval)
  - `data/Local_distilled/distilled_invar_train.jsonl` (400 train, Qwen-27B)
  - `data/Local_distilled/distilled_invar_eval.jsonl` (100 eval)
* **Gold Ground Truth**: `C:/dev/Invar/data/targets/ikuai8.com/gold_set_annotated_v1.jsonl` (60 cards)

---

## 6. Current Objective & Next Single Action

**CURRENT PHASE:** Phase 4 - Dual-Teacher Ensemble & Small Ablation

**CURRENT OBJECTIVE:**
Evaluate teacher consensus (Local Qwen-27B vs Cloud Gemini API), design a Teacher Agreement Gate, and prepare a merged, high-signal training dataset for downstream Invar System-2 verification.

**NEXT SINGLE ACTION:**
Run the teacher agreement analysis and ablation pipeline between `data/Local_distilled/` and `data/API_distilled/` to measure consensus rate, discrepancy distribution, and downstream performance impact on Invar.
