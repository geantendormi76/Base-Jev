# AI Development Handoff Specification

> **Document type:** Cross-session engineering state snapshot / recovery contract
> **Project:** base-jev + Invar (Dual-Engine System)
> **Canonical project root:** `C:\dev\base-jev` & `C:\dev\Invar` & `C:\dev\distiller`
> **Handoff path:** `c/HANDOFF.md`
> **Snapshot date:** 2026-09-21
> **Document version:** 6.0.0 (Grand Breakthrough Edition: Neural Recall 0% -> 80% & Zero-Contamination Verified)
> **Primary purpose:** A new AI must be able to reconstruct the current engineering/research state from this document plus the current repository source, tests, configuration, and artifacts, without relying on the old conversation.

---

## 0. Handoff Metadata

| Item | Value | Level | Notes |
|---|---|---|---|
| Project | base-jev + Invar | [FACT] | System-1 Fast Sensor (23ms) + System-2 Deep Verification |
| Model Factory Root | `C:\dev\base-jev` | [FACT] | Discriminative SFT Factory & ONNX DirectML Runtime |
| Distillation Workshop | `C:\dev\distiller` | [FACT] | Dual-Teacher Distillation (Qwen-27B Local + Gemini API) |
| Execution & Verification | `C:\dev\Invar` | [FACT] | AST Parsing, DualTrack Comparator, Gold Set Evaluator |
| Current Backbone | `models/Qwen3-0.6B-Base` | [FACT] | 1024D Hidden Size, `lm_head` physical stripped |
| Production ONNX SHA256 | `908db8140ab25a32...` | [FACT] | 1251.11 MB single-file FP16 static graph |
| Gold Set Ground Truth | 60 cards, 10 true high-risk | [FACT] | Zero-contamination verified (`Train ∩ Gold = ∅`) |
| **Model Recall on Gold Set** | **80.00% (TP=8/10, FN=2)** | [FACT] | Precision: 36.36%, F1: 0.5000 (Crushed Rule 50% / 0.3448) |
| **Pool B Discrepancy Recall** | **100.00% (TP=5/5, FN=0)** | [FACT] | Precision: 62.50%, F1: 0.7692 (Rule completely failed 0%) |

---

## 1. Project Identity

[DECISION]
The system is an industrial Triad Architecture:
1. **distiller**: Expert knowledge distillation workshop. Employs dual teachers (Local Qwen-27B + Cloud Gemini Flash) with Tree-sitter AST and physical Gold Firewall.
2. **base-jev**: System-1 cognitive reflex frontend. Discriminative SFT (Soft-CE + Brier + Entropy Loss). DirectML FP16 TensorCore execution (~23.08ms). Pure tensor classification, zero generative token overhead.
3. **Invar**: System-2 execution and verification. DualTrack Comparator, IDOR differential testing, Bounded symbolic execution, and future Cost-Aware GRPO.

---

## 2. Historic Milestones Achieved (Phase 1 ~ Phase 3 Complete)

1. **Eliminated Fake 380% Recall Bug**: Refactored evaluator `evaluate_gold_set_v1.py` to rigorous binary confusion matrix ($TP/FP/FN/TN$), Fail-Closed assertion on missing joins, and JSON artifact reporting.
2. **Physical Gold Firewall (Zero Contamination)**: Implemented 10-layer surface boundary firewall in PRism-curator, blocking all 60 Gold Cards. E3 Audit Verified: Surface Overlap = 0, Route Overlap = 0.
3. **Dual-Teacher Distillation & Symmetry**:
   * Local 27B: 500 samples in `distiller/data/Local_distilled/` (20,135s run).
   * Cloud Gemini API: 500 samples in `distiller/data/API_distilled/` (Auth keys, multi-channel failover, live dynamic probing, checkpoint resume).
4. **Broke Prior Collapse (0% -> 80% Recall)**:
   * Replaced dry-run mean mock data (which collapsed to 2.33) with genuine API distillation (27.1% Impact >= 3.5, 29.4% Sensitivity >= 3.5).
   * Fixed Train-Serving Parity (`host` attribute injection and `[:60]` query alignment).
   * High-risk recall surged from **0.0% to 80.00%**, capturing 8 out of 10 true critical vulnerabilities.
   * Pool B (rule blind zone) achieved **100% recall (5/5)** and **0.7692 F1 score**.

---

## 3. Current Objective

[FACT]
Phase 2 (Contract Repair) and Phase 3 (Baseline Discriminative SFT Ignition) are **COMPLETED WITH DISTINCTION**.
The model is proven to have robust out-of-distribution semantic intuition.

**Current Objective**: Phase 4 - Dual-Teacher Ensemble & Small Ablation, preparing for downstream Invar System-2 Integration.

---

## 4. Next Single Action

**CURRENT OBJECTIVE → NEXT SINGLE ACTION**

**当前唯一下一动作：**
执行双师共识融合消融实验（Teacher Agreement Gate / Ensemble）：对比纯本地 27B 权重与纯云端 API 权重在 Invar 上的表现差异，或加权融合两套标签生成最终黄金综合数据集。

---

## 5. Current Scope & Out of Scope

**当前允许操作：**
* `base-jev` recipes, training hyperparameters, and ablation comparisons.
* `distiller` dataset merging and teacher agreement filtering.

**当前阶段明确禁止触碰：**
* ❌ 破坏 `Train ∩ Gold = ∅` 防火墙。
* ❌ 在 `base-jev` 引入自回归文本生成或 GRPO（范畴谬误，属于 Invar System-2）。
* ❌ 允许评估器发生静默 3.0 兜底（必须保持 Fail-Closed）。

---

## 6. Last Known Good State

* **Date**: 2026-09-21
* **Backbone**: `models/Qwen3-0.6B-Base`
* **Checkpoint**: `models/Qwen3-0.6B-Base-SystemOne-Invar`
* **ONNX Artifact**: `models/system_one_unified.onnx` (SHA256: `908db8140ab2...`)
* **Hardware Benchmark**: RTX 3060 DirectML steady latency **23.08 ms** (P99: 24.53 ms)
* **Gold Set Evaluation**:
  * Rule: Precision 26.32%, Recall 50.00%, F1 0.3448
  * Neural: Precision 36.36%, Recall 80.00%, F1 0.5000
  * Pool B (Rule Blind Zone): Neural TP=5/5, Recall 100.0%, F1 0.7692

---

## 7. Data Contracts & Layout

* `distiller/data/API_distilled/`: 400 train, 100 eval (Gemini Flash Auth keys).
* `distiller/data/Local_distilled/`: 400 train, 100 eval (Local Qwen-27B).
* `base-jev/data/API_distilled/`: Active training input.
* `base-jev/data/archive_legacy/`: Isolated legacy artifacts and contaminated replay buffers.
* `Invar/data/targets/ikuai8.com/gold_set_annotated_v1.jsonl`: Sealed 60 Gold Cards.
* `Invar/tmp/gold_set_eval_report.json`: Standardized machine-readable confusion matrix artifact.

---

## 8. Invariants

* **I-01**: `TRAIN ∩ GOLD = ∅` (Surface overlap MUST be 0).
* **I-02**: `len(predictions) == len(endpoints)` (1221/1221 1:1 aligned).
* **I-03**: `base-jev` outputs tensors, not text.
* **I-04**: Join mismatches in evaluator must Fail-Fast (Zero silent 3.0 fallbacks).
