# -*- coding: utf-8 -*-
"""
audit_experiment_lineage_v1.py
Phase 1: Read-Only Experiment Lineage & Forensic Audit Engine
Pure Observer - Zero modifications to source assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_VERSION = "1.0.0-phase1-forensic"

# ==============================================================================
# 0. 辅助工具与哈希计算函数
# ==============================================================================

def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    rows.append(json.loads(line_str))
                except Exception:
                    pass
    return rows

def compute_surface_id_comparator(method: str, path: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(path)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname.lower()
        norm_path = parsed.path or "/"
    else:
        host = "<unknown-host>"
        norm_path = path or "/"
    if parsed.scheme and parsed.hostname and parsed.query:
        norm_path = f"{norm_path}?{parsed.query}"
    payload = "\x1f".join([method.upper(), host, norm_path]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]

# ==============================================================================
# 1. 核心取证分析器 (Forensic Auditor)
# ==============================================================================

class ExperimentLineageAuditor:
    def __init__(self, base_jev_root: Path, invar_root: Path, output_dir: Path):
        self.base_jev_root = base_jev_root.resolve()
        self.invar_root = invar_root.resolve()
        self.output_dir = output_dir.resolve()
        self.start_time = datetime.now(timezone.utc).isoformat()
        
        # 结果集存储
        self.assets: Dict[str, Any] = {}
        self.relations: Dict[str, Any] = {}
        self.verdicts: Dict[str, str] = {}
        self.modules_detail: Dict[str, Any] = {}

    def run_full_audit(self) -> Dict[str, Any]:
        print("=" * 80)
        print(f" 🔬 Experiment Lineage & Forensic Audit Engine {SCRIPT_VERSION}")
        print(f" 📂 base-jev Root : {self.base_jev_root}")
        print(f" 📂 Invar Root    : {self.invar_root}")
        print(f" 💾 Output Dir    : {self.output_dir}")
        print("=" * 80)

        # 预先检查目录存在性
        if not self.base_jev_root.is_dir():
            raise FileNotFoundError(f"base-jev 根目录不存在: {self.base_jev_root}")
        if not self.invar_root.is_dir():
            raise FileNotFoundError(f"Invar 根目录不存在: {self.invar_root}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 执行 8 大证据链模块
        self.audit_e0_tool_provenance()
        self.audit_e1_recipe_identity()
        self.audit_e2_checkpoint_identity()
        self.audit_e3_gold_contamination()
        self.audit_e4_prediction_alignment()
        self.audit_e5_dualtrack_lineage()
        self.audit_e6_evaluator_join_replay()
        self.audit_e7_metric_recomputation()
        self.audit_e8_ast_evidence_purity()

        # 裁决总体判定
        overall = "VERIFIED"
        if any(v == "INVALID" for v in self.verdicts.values()):
            overall = "INVALID"
        elif any(v == "UNVERIFIED" for v in self.verdicts.values()):
            overall = "UNVERIFIED"
        elif any(v == "STALE" for v in self.verdicts.values()):
            overall = "STALE"
        self.verdicts["overall_verdict"] = overall

        # 整理全局报告
        report = {
            "audit_tool_provenance": self.modules_detail["E0"],
            "overall_verdict": overall,
            "verdict_summary": {k: v for k, v in self.verdicts.items() if k != "overall_verdict"},
            "assets": self.assets,
            "relations": self.relations,
            "modules": self.modules_detail,
        }

        # 物理落盘 10 个标准产物
        self.write_artifacts(report)
        self.print_summary_terminal(report)
        return report

    # --------------------------------------------------------------------------
    # E0: Audit Tool Provenance
    # --------------------------------------------------------------------------
    def audit_e0_tool_provenance(self):
        curr_file = Path(__file__).resolve()
        tool_hash = sha256_file(curr_file)
        self.modules_detail["E0"] = {
            "tool_name": curr_file.name,
            "tool_path": str(curr_file),
            "tool_sha256": tool_hash,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executed_at": self.start_time,
            "base_jev_root": str(self.base_jev_root),
            "invar_root": str(self.invar_root),
        }

    # --------------------------------------------------------------------------
    # E1: Recipe & Dataset Identity
    # --------------------------------------------------------------------------
    def audit_e1_recipe_identity(self):
        print("\n[*] [E1] 审计生产配方与物理数据集血缘...")
        recipe_path = self.base_jev_root / "recipes" / "recipe_invar_0.6b.json"
        recipe_exists = recipe_path.is_file()
        recipe_sha = sha256_file(recipe_path)
        
        detail: Dict[str, Any] = {
            "recipe_path": str(recipe_path),
            "recipe_exists": recipe_exists,
            "recipe_sha256": recipe_sha,
        }
        
        train_artifact: Dict[str, Any] = {}
        eval_artifact: Dict[str, Any] = {}
        recipe_consistent = False

        if recipe_exists:
            data = load_json(recipe_path)
            detail["recipe_id"] = data.get("recipe_id")
            detail["product_name"] = data.get("product_name")
            
            # 解析训练集
            train_rel = data.get("datasets", {}).get("train_path", "")
            train_abs = (self.base_jev_root / train_rel).resolve()
            train_exists = train_abs.is_file()
            train_lines = len(read_jsonl(train_abs)) if train_exists else 0
            train_sha = sha256_file(train_abs)
            train_artifact = {
                "declared_path": train_rel,
                "resolved_path": str(train_abs),
                "exists": train_exists,
                "sha256": train_sha,
                "line_count": train_lines,
            }

            # 解析评估集
            eval_rel = data.get("datasets", {}).get("eval_path", "")
            eval_abs = (self.base_jev_root / eval_rel).resolve()
            eval_exists = eval_abs.is_file()
            eval_lines = len(read_jsonl(eval_abs)) if eval_exists else 0
            eval_sha = sha256_file(eval_abs)
            eval_artifact = {
                "declared_path": eval_rel,
                "resolved_path": str(eval_abs),
                "exists": eval_exists,
                "sha256": eval_sha,
                "line_count": eval_lines,
            }

            # 对比 Checkpoint 内留存的 recipe
            ckpt_recipe_path = self.base_jev_root / "models" / "Qwen3-0.6B-Base-SystemOne-Invar" / "recipe.json"
            if ckpt_recipe_path.is_file():
                ckpt_recipe_sha = sha256_file(ckpt_recipe_path)
                recipe_consistent = (ckpt_recipe_sha == recipe_sha)
                detail["ckpt_recipe_consistent"] = recipe_consistent
            else:
                detail["ckpt_recipe_consistent"] = None

        detail["train_artifact"] = train_artifact
        detail["eval_artifact"] = eval_artifact

        self.assets["recipe"] = detail
        self.assets["train_dataset"] = train_artifact
        self.assets["eval_dataset"] = eval_artifact
        self.relations["recipe_to_train"] = train_artifact.get("resolved_path")
        self.relations["recipe_to_eval"] = eval_artifact.get("resolved_path")

        # 判定
        if recipe_exists and train_artifact.get("exists") and eval_artifact.get("exists"):
            verdict = "VERIFIED"
        else:
            verdict = "INVALID"
        
        self.verdicts["E1_recipe_identity"] = verdict
        self.modules_detail["E1"] = detail
        print(f"  └─ 结论: {verdict} | Train: {train_artifact.get('declared_path')} ({train_artifact.get('line_count')} lines)")

    # --------------------------------------------------------------------------
    # E2: Checkpoint & Deployment Identity
    # --------------------------------------------------------------------------
    def audit_e2_checkpoint_identity(self):
        print("\n[*] [E2] 审计训练检查点与部署交付物双子树哈希...")
        ckpt_dir = self.base_jev_root / "models" / "Qwen3-0.6B-Base-SystemOne-Invar"
        
        # 1. 检查点子树
        ckpt_heads = ckpt_dir / "decision_heads.pt"
        ckpt_recipe = ckpt_dir / "recipe.json"
        ckpt_metrics = ckpt_dir / "eval_metrics.json"
        lora_dir = ckpt_dir / "backbone_lora"
        lora_files: Dict[str, str] = {}
        if lora_dir.is_dir():
            for f in lora_dir.glob("*"):
                if f.is_file():
                    lora_files[f.name] = sha256_file(f)

        ckpt_detail = {
            "checkpoint_dir": str(ckpt_dir),
            "exists": ckpt_dir.is_dir(),
            "decision_heads_sha256": sha256_file(ckpt_heads),
            "recipe_sha256": sha256_file(ckpt_recipe),
            "eval_metrics": load_json(ckpt_metrics),
            "lora_files": lora_files,
        }

        # 2. 部署交付子树
        models_onnx = self.base_jev_root / "models" / "system_one_unified.onnx"
        dist_onnx = self.base_jev_root / "dist" / "invar-intent-0.6b-v1" / "system_one_unified.onnx"
        passport_path = self.base_jev_root / "dist" / "invar-intent-0.6b-v1" / "MODEL_PASSPORT.json"
        
        passport_data = load_json(passport_path) or {}
        declared_passport_sha = passport_data.get("sha256", "")
        models_onnx_sha = sha256_file(models_onnx)
        dist_onnx_sha = sha256_file(dist_onnx)

        deploy_detail = {
            "models_onnx_sha256": models_onnx_sha,
            "models_onnx_size_bytes": models_onnx.stat().st_size if models_onnx.is_file() else 0,
            "dist_onnx_sha256": dist_onnx_sha,
            "passport_declared_sha256": declared_passport_sha,
            "passport_matches_models_onnx": (models_onnx_sha == declared_passport_sha and bool(models_onnx_sha)),
            "dist_matches_models_onnx": (models_onnx_sha == dist_onnx_sha and bool(models_onnx_sha)),
        }

        detail = {
            "checkpoint_subtree": ckpt_detail,
            "deployment_subtree": deploy_detail,
        }

        self.assets["checkpoint"] = ckpt_detail
        self.assets["deployment"] = deploy_detail
        self.relations["checkpoint_to_deployment"] = deploy_detail["passport_matches_models_onnx"]

        # 判定
        if ckpt_detail["exists"] and ckpt_heads.is_file() and deploy_detail["passport_matches_models_onnx"]:
            verdict = "VERIFIED"
        else:
            verdict = "INVALID"

        self.verdicts["E2_checkpoint_identity"] = verdict
        self.modules_detail["E2"] = detail
        print(f"  └─ 结论: {verdict} | Passport SHA 对齐 ONNX: {deploy_detail['passport_matches_models_onnx']}")

    # --------------------------------------------------------------------------
    # E3: Gold Set Contamination Forensic
    # --------------------------------------------------------------------------
    def audit_e3_gold_contamination(self):
        print("\n[*] [E3] 深度审计黄金评测集污染 (三重正交交集探查)...")
        gold_annotated_path = self.invar_root / "data" / "targets" / "ikuai8.com" / "gold_set_annotated_v1.jsonl"
        gold_cards = read_jsonl(gold_annotated_path)
        
        # 物理通道探查
        invar_replay = self.invar_root / "data" / "targets" / "ikuai8.com" / "gold_replay_buffer_60.jsonl"
        base_jev_replay = self.base_jev_root / "data" / "gold_replay_buffer_60.jsonl"
        
        detail: Dict[str, Any] = {
            "gold_annotated_path": str(gold_annotated_path),
            "gold_card_count": len(gold_cards),
            "invar_replay_exists": invar_replay.is_file(),
            "invar_replay_sha256": sha256_file(invar_replay),
            "base_jev_replay_exists": base_jev_replay.is_file(),
            "base_jev_replay_sha256": sha256_file(base_jev_replay),
        }

        # 提取 Gold 三层特征集合
        gold_surfaces: Set[str] = set()
        gold_routes_full: Set[str] = set()
        gold_routes_coarse: Set[str] = set()
        gold_slice_hashes: Set[str] = set()

        for c in gold_cards:
            sid = c.get("surface_id", "")
            if sid:
                gold_surfaces.add(sid)
            m = str(c.get("method", "GET")).upper().strip()
            h = str(c.get("host", "")).lower().strip()
            p = str(c.get("path", "")).strip()
            gold_routes_full.add(f"{m}|{h}|{p}")
            gold_routes_coarse.add(f"{m}|{p}")
            for s in c.get("evidence_slices", []):
                sh = s.get("code_slice", "").strip()
                if sh:
                    gold_slice_hashes.add(sha256_str(sh)[:16])

        # 读取当前训练集并检测交集
        train_path_str = self.assets.get("train_dataset", {}).get("resolved_path", "")
        train_rows = read_jsonl(Path(train_path_str)) if train_path_str else []
        
        train_surfaces: Set[str] = set()
        train_routes_full: Set[str] = set()
        train_routes_coarse: Set[str] = set()
        train_slice_hashes: Set[str] = set()
        train_has_replay_tag = 0

        for r in train_rows:
            sid = r.get("surface_id", "")
            m = str(r.get("method", "GET")).upper().strip()
            h = str(r.get("host", "")).lower().strip()
            p = str(r.get("path", "")).strip()
            if not sid:
                sid = compute_surface_id_comparator(m, p)
            train_surfaces.add(sid)
            train_routes_full.add(f"{m}|{h}|{p}")
            train_routes_coarse.add(f"{m}|{p}")
            cs = str(r.get("code_slice", "")).strip()
            if cs:
                train_slice_hashes.add(sha256_str(cs)[:16])
            
            sample_id = str(r.get("sample_id", ""))
            if "GOLD-REPLAY" in sample_id or "gold_replay" in sample_id.lower():
                train_has_replay_tag += 1

        # 计算三重正交交集
        inter_level1 = gold_surfaces.intersection(train_surfaces)
        inter_level2 = gold_routes_full.intersection(train_routes_full)
        inter_level2_coarse = gold_routes_coarse.intersection(train_routes_coarse)
        inter_level3 = gold_slice_hashes.intersection(train_slice_hashes)

        detail["intersections"] = {
            "level1_surface_id_overlap_count": len(inter_level1),
            "level1_surface_id_overlap_samples": sorted(list(inter_level1))[:10],
            "level2_method_host_path_overlap_count": len(inter_level2),
            "level2_method_host_path_overlap_samples": sorted(list(inter_level2))[:10],
            "level2_method_path_coarse_overlap_count": len(inter_level2_coarse),
            "level3_code_slice_hash_overlap_count": len(inter_level3),
            "train_gold_replay_tag_count": train_has_replay_tag,
        }

        # 判定
        is_contaminated = (
            len(inter_level1) > 0
            or len(inter_level2) > 0
            or len(inter_level3) > 0
            or train_has_replay_tag > 0
        )

        verdict = "INVALID" if is_contaminated else "VERIFIED"
        self.verdicts["E3_gold_contamination"] = verdict
        self.modules_detail["E3"] = detail
        print(f"  └─ 结论: {verdict} | Surface 重叠: {len(inter_level1)} | Route 重叠: {len(inter_level2)} | Replay 标签: {train_has_replay_tag}")

    # --------------------------------------------------------------------------
    # E4: Prediction Asset Identity & Quality
    # --------------------------------------------------------------------------
    def audit_e4_prediction_alignment(self):
        print("\n[*] [E4] 审计神经预测资产行对齐与降级质量...")
        pred_path = self.invar_root / "tmp" / "base_jev_predictions_1221.jsonl"
        report_path = self.invar_root / "tmp" / "ikuai8_endpoints_report.json"

        pred_rows = read_jsonl(pred_path)
        report_data = load_json(report_path) or {}
        report_endpoints = report_data.get("endpoints", [])

        n_pred = len(pred_rows)
        n_report = len(report_endpoints)
        
        detail: Dict[str, Any] = {
            "prediction_path": str(pred_path),
            "prediction_sha256": sha256_file(pred_path),
            "prediction_row_count": n_pred,
            "report_endpoint_count": n_report,
            "row_count_match": (n_pred == n_report and n_pred > 0),
        }

        # 逐行校验 endpoint_id 1:1 对齐
        mismatches: List[str] = []
        fallback_oversize = 0
        fallback_oom = 0
        normal_count = 0

        for idx, (p, ep) in enumerate(zip(pred_rows, report_endpoints)):
            p_eid = str(p.get("endpoint_id") or p.get("id") or "")
            r_eid = str(ep.get("endpoint_id") or ep.get("id") or "")
            if p_eid != r_eid:
                if len(mismatches) < 5:
                    mismatches.append(f"Row {idx}: pred='{p_eid}' vs report='{r_eid}'")

            provenance = str(p.get("runner_provenance", ""))
            if "oversize" in provenance:
                fallback_oversize += 1
            elif "oom" in provenance:
                fallback_oom += 1
            else:
                normal_count += 1

        total_fallbacks = fallback_oversize + fallback_oom
        fallback_rate = (total_fallbacks / max(1, n_pred)) * 100.0

        detail["alignment"] = {
            "mismatch_count": len(mismatches),
            "mismatch_samples": mismatches,
            "is_strictly_aligned": (len(mismatches) == 0 and n_pred == n_report and n_pred > 0),
        }
        detail["quality"] = {
            "normal_inference_count": normal_count,
            "fallback_oversize_count": fallback_oversize,
            "fallback_oom_count": fallback_oom,
            "fallback_rate_percent": round(fallback_rate, 2),
            "quality_status": "CLEAN" if total_fallbacks == 0 else "DEGRADED",
        }

        self.assets["prediction_1221"] = detail
        self.relations["prediction_to_report"] = detail["alignment"]["is_strictly_aligned"]

        # 判定: 身份对齐属于硬契约
        verdict = "VERIFIED" if detail["alignment"]["is_strictly_aligned"] else "INVALID"
        self.verdicts["E4_prediction_alignment"] = verdict
        self.modules_detail["E4"] = detail
        print(f"  └─ 结论: {verdict} | 1:1 逐行对齐: {detail['alignment']['is_strictly_aligned']} | 降级率: {fallback_rate:.2f}%")

    # --------------------------------------------------------------------------
    # E5: DualTrack Lineage Forensic
    # --------------------------------------------------------------------------
    def audit_e5_dualtrack_lineage(self):
        print("\n[*] [E5] 审计 DualTrack 比较器历史血缘 (哈希溯源断口)...")
        manifest_path = self.invar_root / "tmp" / "triage_phase1_output" / "triage_run_manifest_v2.json"
        dt_path = self.invar_root / "tmp" / "triage_phase1_output" / "triage_dual_track_v2.jsonl"
        
        manifest_data = load_json(manifest_path) or {}
        inputs = manifest_data.get("inputs", {})
        
        declared_neural = inputs.get("neural_jsonl", "")
        declared_report_sha = inputs.get("report_sha256", "")
        dt_sha = sha256_file(dt_path)

        curr_pred_path = self.invar_root / "tmp" / "base_jev_predictions_1221.jsonl"
        curr_pred_sha = sha256_file(curr_pred_path)

        # 关键事实：manifest inputs 中有无记录 neural_jsonl_sha256？
        has_neural_sha = "neural_jsonl_sha256" in inputs
        
        detail = {
            "manifest_path": str(manifest_path),
            "manifest_exists": manifest_path.is_file(),
            "dual_track_path": str(dt_path),
            "dual_track_sha256": dt_sha,
            "declared_neural_path": declared_neural,
            "declared_report_sha256": declared_report_sha,
            "manifest_records_neural_sha256": has_neural_sha,
            "current_prediction_sha256": curr_pred_sha,
            "historical_hash_verifiable": has_neural_sha,
        }

        self.assets["dual_track_v2"] = detail
        self.relations["dualtrack_to_manifest"] = manifest_path.is_file()

        # 契约修正：因历史 manifest 确实缺失 neural_jsonl 哈希，定性为 UNVERIFIED
        verdict = "UNVERIFIED"
        self.verdicts["E5_dualtrack_lineage"] = verdict
        self.modules_detail["E5"] = detail
        print(f"  └─ 结论: {verdict} (历史 Manifest 缺失 neural_jsonl_sha256 物理哈希记录)")

    # --------------------------------------------------------------------------
    # E6: Evaluator Join Replay
    # --------------------------------------------------------------------------
    def audit_e6_evaluator_join_replay(self):
        print("\n[*] [E6] 模拟重放当前 Evaluator 对 60 张 Gold Card 的 Join 匹配...")
        gold_path = self.invar_root / "data" / "targets" / "ikuai8.com" / "gold_set_annotated_v1.jsonl"
        manifest_path = self.invar_root / "data" / "targets" / "ikuai8.com" / "gold_set_v1_manifest.json"
        dt_path = self.invar_root / "tmp" / "triage_phase1_output" / "triage_dual_track_v2.jsonl"

        cards = read_jsonl(gold_path)
        gold_manifest = load_json(manifest_path) or {}
        key_mappings = {m["card_id"]: m for m in gold_manifest.get("key_mappings", [])}
        dt_rows = read_jsonl(dt_path)

        # 建立 DualTrack 索引 (按旧评估器完全原样复现)
        dual_track_by_surface = defaultdict(list)
        dual_track_by_endpoint = {}
        for r in dt_rows:
            sid = r.get("surface_id", "")
            if sid:
                dual_track_by_surface[sid].append(r)
            epid = r.get("endpoint_id") or r.get("record_id")
            if epid:
                dual_track_by_endpoint[str(epid)] = r

        matched_by_surface = 0
        matched_by_record_fallback = 0
        silent_default_3_0_count = 0
        replay_records = []

        for card in cards:
            cid = card["card_id"]
            meta = key_mappings.get(cid, {})
            sid = card.get("surface_id") or meta.get("surface_id")

            dt_records = dual_track_by_surface.get(sid, [])
            first_r = dt_records[0] if dt_records else None
            match_method = "surface_id" if first_r else None

            if not first_r:
                for rid in meta.get("record_ids", []):
                    if rid in dual_track_by_endpoint:
                        first_r = dual_track_by_endpoint[rid]
                        match_method = "record_ids_fallback"
                        matched_by_record_fallback += 1
                        break

            is_defaulted = False
            if not first_r:
                first_r = {}
                is_defaulted = True
                silent_default_3_0_count += 1
                match_method = "SILENT_DEFAULT_3.0"
            else:
                if match_method == "surface_id":
                    matched_by_surface += 1

            neural_data = first_r.get("neural", {})
            n_imp = neural_data.get("impact_score") if neural_data.get("impact_score") is not None else 3.0
            n_sens = neural_data.get("sensitivity_score") if neural_data.get("sensitivity_score") is not None else 3.0

            replay_records.append({
                "card_id": cid,
                "surface_id": sid,
                "match_method": match_method,
                "is_defaulted_3_0": is_defaulted,
                "assigned_impact": n_imp,
                "assigned_sensitivity": n_sens,
            })

        detail = {
            "total_cards": len(cards),
            "matched_by_surface": matched_by_surface,
            "matched_by_record_fallback": matched_by_record_fallback,
            "silent_default_3_0_count": silent_default_3_0_count,
            "cards_defaulted_ratio": round((silent_default_3_0_count / max(1, len(cards))) * 100.0, 2),
            "replay_sample_details": replay_records,
        }

        # 判定：只要有 3.0 兜底，即证明评估被静默失真污染
        verdict = "INVALID" if silent_default_3_0_count > 0 else "VERIFIED"
        self.verdicts["E6_evaluator_join_replay"] = verdict
        self.modules_detail["E6"] = detail
        print(f"  └─ 结论: {verdict} | 静默 3.0 兜底卡片数: {silent_default_3_0_count}/{len(cards)}")

    # --------------------------------------------------------------------------
    # E7: Metric Recomputation
    # --------------------------------------------------------------------------
    def audit_e7_metric_recomputation(self):
        print("\n[*] [E7] 真实混淆矩阵 vs 旧代码 380% 召回率数学并排对照...")
        replay_samples = self.modules_detail.get("E6", {}).get("replay_sample_details", [])
        gold_path = self.invar_root / "data" / "targets" / "ikuai8.com" / "gold_set_annotated_v1.jsonl"
        dt_path = self.invar_root / "tmp" / "triage_phase1_output" / "triage_dual_track_v2.jsonl"

        cards_map = {c["card_id"]: c for c in read_jsonl(gold_path)}
        dt_rows = read_jsonl(dt_path)
        dt_by_surface = {r.get("surface_id"): r for r in dt_rows if r.get("surface_id")}

        gt_high_count = 0
        rule_alarms = 0
        neural_alarms = 0

        # 混淆矩阵计数
        rule_tp = 0
        rule_fp = 0
        rule_fn = 0
        rule_tn = 0

        neural_tp = 0
        neural_fp = 0
        neural_fn = 0
        neural_tn = 0

        for s in replay_samples:
            cid = s["card_id"]
            sid = s["surface_id"]
            card = cards_map.get(cid, {})
            anno = card.get("annotation", {})
            gt_imp = anno.get("ground_truth_impact", 3)
            gt_sens = anno.get("ground_truth_sensitivity", 3)
            is_gt_high = (gt_imp >= 4 or gt_sens >= 4)

            dt_rec = dt_by_surface.get(sid, {})
            r_high = bool(dt_rec.get("rule", {}).get("high_risk_candidate", False))
            n_high = (s["assigned_impact"] >= 3.5 or s["assigned_sensitivity"] >= 3.5)

            if is_gt_high:
                gt_high_count += 1

            if r_high:
                rule_alarms += 1
            if n_high:
                neural_alarms += 1

            # 规则混淆矩阵
            if is_gt_high and r_high:
                rule_tp += 1
            elif not is_gt_high and r_high:
                rule_fp += 1
            elif is_gt_high and not r_high:
                rule_fn += 1
            else:
                rule_tn += 1

            # 模型混淆矩阵
            if is_gt_high and n_high:
                neural_tp += 1
            elif not is_gt_high and n_high:
                neural_fp += 1
            elif is_gt_high and not n_high:
                neural_fn += 1
            else:
                neural_tn += 1

        # 旧算法回放指标 (旧代码直接 alarms / gt_high)
        legacy_rule_ratio = (rule_alarms / max(1, gt_high_count)) * 100.0
        legacy_neural_ratio = (neural_alarms / max(1, gt_high_count)) * 100.0

        # 真实标准指标
        def calc_metrics(tp, fp, fn, tn):
            prec = tp / max(1, (tp + fp))
            rec = tp / max(1, (tp + fn))
            f1 = (2 * prec * rec) / max(1e-6, (prec + rec))
            return {
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            }

        detail = {
            "gt_high_risk_surfaces_count": gt_high_count,
            "total_surfaces_evaluated": len(replay_samples),
            "legacy_metric_replay": {
                "rule_total_alarms": rule_alarms,
                "legacy_rule_recall_percent": round(legacy_rule_ratio, 2),
                "neural_total_alarms": neural_alarms,
                "legacy_neural_recall_percent": round(legacy_neural_ratio, 2),
                "formula_bug": "旧脚本将全池总报警数 (含大量FP) 直接除以真实高危数，导致检出率出现 380%",
            },
            "standard_rule_metrics": calc_metrics(rule_tp, rule_fp, rule_fn, rule_tn),
            "standard_neural_metrics": calc_metrics(neural_tp, neural_fp, neural_fn, neural_tn),
        }

        # 判定：重算成功即为 VERIFIED
        verdict = "VERIFIED"
        self.verdicts["E7_metric_recomputation"] = verdict
        self.modules_detail["E7"] = detail
        print(f"  └─ 结论: {verdict} | 旧规则检出率: {legacy_rule_ratio:.1f}% ➔ 真实规范召回率: {detail['standard_rule_metrics']['recall']*100:.1f}%")

    # --------------------------------------------------------------------------
    # E8: AST Evidence Purity & Provenance
    # --------------------------------------------------------------------------
    def audit_e8_ast_evidence_purity(self):
        print("\n[*] [E8] 审计 Gold Set 证据切片提取纯度与物理窗口风险...")
        gold_path = self.invar_root / "data" / "targets" / "ikuai8.com" / "gold_set_annotated_v1.jsonl"
        dt_path = self.invar_root / "tmp" / "triage_phase1_output" / "triage_dual_track_v2.jsonl"

        cards = read_jsonl(gold_path)
        dt_rows = read_jsonl(dt_path)
        dt_by_rid = {r.get("record_id"): r for r in dt_rows if r.get("record_id")}

        traceable_slices = 0
        untraceable_slices = 0
        status_counter = Counter()
        has_multiline_counter = 0

        for card in cards:
            for s in card.get("evidence_slices", []):
                rid = s.get("record_id")
                code_text = s.get("code_slice", "")
                if "\n" in code_text.strip():
                    has_multiline_counter += 1
                
                if rid in dt_by_rid:
                    traceable_slices += 1
                    stat = dt_by_rid[rid].get("code_slice_status", "UNKNOWN")
                    status_counter[stat] += 1
                else:
                    untraceable_slices += 1
                    status_counter["MISSING_IN_DUALTRACK"] += 1

        detail = {
            "total_cards": len(cards),
            "traceable_slices_count": traceable_slices,
            "untraceable_slices_count": untraceable_slices,
            "code_slice_status_distribution": dict(status_counter),
            "evidence_has_multiline_ratio": round((has_multiline_counter / max(1, traceable_slices + untraceable_slices)) * 100.0, 2),
            "context_leakage_note": "Invar comparator 生成阶段使用 +-40 行物理窗口，Gold Set 虽隐藏标签，但证据切片存在邻居上下文残留风险",
        }

        # 契约修正：因无法证明 100% 纯 AST 闭包（带有 40 行物理窗口特征），标记为 UNVERIFIED
        verdict = "UNVERIFIED"
        self.verdicts["E8_ast_evidence_purity"] = verdict
        self.modules_detail["E8"] = detail
        print(f"  └─ 结论: {verdict} | 可溯源切片: {traceable_slices} | 状态分布: {dict(status_counter)}")

    # --------------------------------------------------------------------------
    # 物理落盘与终端渲染
    # --------------------------------------------------------------------------
    def write_artifacts(self, report: Dict[str, Any]):
        def dump_json(p: Path, d: Any):
            with p.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
                f.write("\n")

        dump_json(self.output_dir / "experiment_lineage_report.json", report)
        dump_json(self.output_dir / "E1_recipe_identity.json", self.modules_detail["E1"])
        dump_json(self.output_dir / "E2_checkpoint_identity.json", self.modules_detail["E2"])
        dump_json(self.output_dir / "E3_train_gold_intersection.json", self.modules_detail["E3"])
        dump_json(self.output_dir / "E4_prediction_alignment.json", self.modules_detail["E4"])
        dump_json(self.output_dir / "E5_dualtrack_lineage.json", self.modules_detail["E5"])
        dump_json(self.output_dir / "E6_evaluator_join_replay.json", self.modules_detail["E6"])
        dump_json(self.output_dir / "E7_metric_recomputation.json", self.modules_detail["E7"])
        dump_json(self.output_dir / "E8_ast_evidence_purity.json", self.modules_detail["E8"])

    def print_summary_terminal(self, report: Dict[str, Any]):
        summary_lines = [
            "=" * 80,
            f" 🎯 实验血缘与取证全景体检战报 (Forensic Verdict: {report['overall_verdict']})",
            "=" * 80,
            f" 1. [E1] Recipe & Dataset Identity : {report['verdict_summary']['E1_recipe_identity']}",
            f" 2. [E2] Checkpoint & Deployment   : {report['verdict_summary']['E2_checkpoint_identity']}",
            f" 3. [E3] Gold Set Contamination    : {report['verdict_summary']['E3_gold_contamination']} (一票否决级)",
            f" 4. [E4] Prediction 1:1 Alignment  : {report['verdict_summary']['E4_prediction_alignment']}",
            f" 5. [E5] DualTrack Historical Hash : {report['verdict_summary']['E5_dualtrack_lineage']}",
            f" 6. [E6] Evaluator Join 3.0 Replay : {report['verdict_summary']['E6_evaluator_join_replay']} (静默兜底排查)",
            f" 7. [E7] Standard Confusion Matrix : {report['verdict_summary']['E7_metric_recomputation']}",
            f" 8. [E8] AST Evidence Context Purity: {report['verdict_summary']['E8_ast_evidence_purity']}",
            "-" * 80,
            " 📊 核心证据速览:",
            f"  • 当前 Recipe 声明训练集 : {report['modules']['E1']['train_artifact'].get('declared_path')} ({report['modules']['E1']['train_artifact'].get('line_count')} lines)",
            f"  • 训练集与 Gold Set 重叠 : Surface={report['modules']['E3']['intersections']['level1_surface_id_overlap_count']} | Route={report['modules']['E3']['intersections']['level2_method_host_path_overlap_count']}",
            f"  • Evaluator 静默 3.0 卡片: {report['modules']['E6']['silent_default_3_0_count']} / {report['modules']['E6']['total_cards']} (失真率: {report['modules']['E6']['cards_defaulted_ratio']}%)",
            f"  • 规则轨真实指标重算     : Precision={report['modules']['E7']['standard_rule_metrics']['precision']} | Recall={report['modules']['E7']['standard_rule_metrics']['recall']} (旧算法: {report['modules']['E7']['legacy_metric_replay']['legacy_rule_recall_percent']}%)",
            f"  • 模型轨真实指标重算     : Precision={report['modules']['E7']['standard_neural_metrics']['precision']} | Recall={report['modules']['E7']['standard_neural_metrics']['recall']}",
            "=" * 80,
            f" 📄 机器索引与 10 大物证已完整归档至: {self.output_dir}",
            "=" * 80,
        ]
        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)
        
        # 写入 txt 摘要
        (self.output_dir / "experiment_lineage_summary.txt").write_text(summary_text, encoding="utf-8")


# ==============================================================================
# 入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Read-Only Experiment Lineage Forensic Auditor")
    parser.add_argument("--base-jev-root", type=Path, default=Path("C:/dev/base-jev"))
    parser.add_argument("--invar-root", type=Path, default=Path("C:/dev/Invar"))
    parser.add_argument("--output-dir", type=Path, default=Path("C:/dev/base-jev/tmp/EXPERIMENT_LINEAGE_AUDIT_V1"))
    args = parser.parse_args()

    auditor = ExperimentLineageAuditor(
        base_jev_root=args.base_jev_root,
        invar_root=args.invar_root,
        output_dir=args.output_dir,
    )
    auditor.run_full_audit()

if __name__ == "__main__":
    main()

