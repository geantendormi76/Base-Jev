import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "python" / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from contracts import (
    BatchDecisionRequest,
    ScoreQuestion,
    TypedStateSnapshot,
)
from sequence_packer import SequencePacker
from system_one_model import SystemOneModel

# 激活 PyTorch 显存碎片管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 装配 Tree-sitter AST 引擎
try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript as tsjavascript
    _HAS_TREE_SITTER = True
    _JS_LANGUAGE = Language(tsjavascript.language())
except Exception:
    _HAS_TREE_SITTER = False
    _JS_LANGUAGE = None


def resolve_path(base_path: Path, p_str: str) -> Path:
    p = Path(p_str)
    return p if p.is_absolute() else (base_path / p).resolve()


class FastAstExtractor:
    def __init__(self, window_chars: int = 250):
        self.window_chars = window_chars
        self._tree_cache: Dict[str, Any] = {}
        if _HAS_TREE_SITTER and _JS_LANGUAGE is not None:
            try:
                self.parser = Parser(_JS_LANGUAGE)
            except TypeError:
                self.parser = Parser()
                self.parser.language = _JS_LANGUAGE
        else:
            self.parser = None

    def _get_tree(self, physical_file: Path, source_bytes: bytes) -> Any:
        file_key = str(physical_file.resolve())
        if file_key in self._tree_cache:
            return self._tree_cache[file_key]
        tree = self.parser.parse(source_bytes)
        self._tree_cache[file_key] = tree
        return tree

    def extract_slice(self, file_p: Optional[Path], path_str: str, call_sig: str) -> str:
        # 异常超长路径物理防御
        safe_path = path_str[:150]
        if file_p is None or not file_p.is_file():
            fallback = call_sig if (call_sig and len(call_sig) >= 6) else safe_path
            return fallback[:self.window_chars].strip()

        try:
            content = file_p.read_text(encoding="utf-8", errors="ignore")
            source_bytes = content.encode("utf-8", errors="ignore")
        except Exception:
            return safe_path[:self.window_chars]

        if self.parser is not None:
            try:
                tree = self._get_tree(file_p, source_bytes)
                root = tree.root_node
                stack = [root]
                while stack:
                    curr = stack.pop()
                    if curr.type == "call_expression":
                        node_text = curr.text.decode("utf-8", errors="ignore")
                        if safe_path in node_text:
                            enclosing = curr
                            parent = curr.parent
                            while parent is not None and parent.type != "program":
                                if parent.type in ("function_declaration", "function_expression", "arrow_function", "method_definition", "export_statement"):
                                    enclosing = parent
                                    break
                                parent = parent.parent

                            enc_text = enclosing.text.decode("utf-8", errors="ignore")
                            return re.sub(r"\s+", " ", enc_text).strip()[:self.window_chars]

                    for ch in reversed(curr.named_children):
                        stack.append(ch)
            except Exception:
                pass

        idx = content.find(f'"{safe_path}"')
        if idx == -1:
            idx = content.find(f"'{safe_path}'")
        if idx == -1:
            idx = content.find(safe_path)

        if idx != -1:
            start = max(0, idx - 80)
            end = min(len(content), idx + len(safe_path) + 120)
            return re.sub(r"\s+", " ", content[start:end]).strip()[:self.window_chars]

        return (call_sig[:self.window_chars] if call_sig else safe_path).strip()


def main():
    parser = argparse.ArgumentParser(description="base-jev: 为 Invar 比较器生成 1221 全量神经预测 JSONL")
    parser.add_argument("--report", default="C:/dev/Invar/tmp/ikuai8_endpoints_report.json", help="Invar 静态扫描报告")
    parser.add_argument("--raw-js", default="C:/dev/Invar/tmp/raw_js", help="Invar 原始 JS 根目录")
    parser.add_argument("--output", default="C:/dev/Invar/tmp/base_jev_predictions_1221.jsonl", help="输出预测目标文件")
    parser.add_argument("--limit", type=int, default=None, help="仅用于小规模测试的条数限制")
    args = parser.parse_args()

    report_p = resolve_path(PROJECT_ROOT, args.report)
    raw_js_p = resolve_path(PROJECT_ROOT, args.raw_js)
    output_p = resolve_path(PROJECT_ROOT, args.output)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" 🚀 base-jev: Invar 双轨比较器专属预测适配引擎 (高鲁棒防御版)")
    print(f" 📂 输入报告: {report_p}")
    print(f" 📂 源码目录: {raw_js_p}")
    print(f" 📦 交付输出: {output_p}")
    print("=" * 75)

    if not report_p.is_file():
        raise FileNotFoundError(f"找不到报告文件: {report_p}")

    with report_p.open("r", encoding="utf-8") as f:
        report_data = json.load(f)

    endpoints = report_data.get("endpoints", [])
    if args.limit:
        endpoints = endpoints[:args.limit]

    total_endpoints = len(endpoints)
    print(f"[*] 待推演端点总数: {total_endpoints} 个")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_dir = PROJECT_ROOT / "models" / "Qwen3-0.6B-Base"
    checkpoint_dir = PROJECT_ROOT / "models" / "Qwen3-0.6B-Base-SystemOne-Invar"

    print(f"[*] 正在挂载 0.6B 纯表征底座与物理双头到 {device.upper()}...")
    model = SystemOneModel.from_pretrained_base(base_model_dir, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    lora_dir = checkpoint_dir / "backbone_lora"
    if lora_dir.is_dir():
        model.backbone = PeftModel.from_pretrained(model.backbone, str(lora_dir))

    heads_file = checkpoint_dir / "decision_heads.pt"
    if heads_file.is_file():
        model.heads.load_state_dict(torch.load(heads_file, map_location=device), strict=False)

    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    packer = SequencePacker(tokenizer)
    ast_extractor = FastAstExtractor(window_chars=250)

    print(f"[*] 启动全量端到端防御性推演 (自动防御超长异常)...")
    results = []
    t_start_all = time.perf_counter()
    abnormal_samples_count = 0

    with torch.inference_mode():
        for idx, ep in enumerate(endpoints, start=1):
            t_sample_start = time.perf_counter()
            endpoint_id = ep.get("endpoint_id") or ep.get("id") or f"EP-{idx:04d}"
            method = str(ep.get("method", "GET")).strip().upper()[:10]
            raw_path = str(ep.get("path", "")).strip()
            
            # 第一道防线：物理限制字符串长度，防巨型 DataURI/SVG 污染
            safe_path = raw_path[:200]
            source_file = ep.get("source_file")
            call_sig = str(ep.get("call_signature", ""))[:150]

            target_file = None
            if source_file and raw_js_p.is_dir():
                cand = raw_js_p / source_file
                if cand.is_file():
                    target_file = cand
                else:
                    found = list(raw_js_p.rglob(Path(source_file).name))
                    target_file = found[0] if found else None

            code_slice = ast_extractor.extract_slice(target_file, safe_path, call_sig)
            
            # 严格限制参数字符串长度
            params = [str(p).strip()[:40] for p in ep.get("extracted_params", []) if str(p).strip()]
            params_str = ",".join(params[:8])

            # 从 source_file 解析 host，确保与训练集 TypedStateSnapshot 绝对同构
            parts = Path(source_file).parts if source_file else []
            host_val = parts[1] if len(parts) > 1 and parts[0] == "raw_js" else "cloud.ikuai8.com"

            state = TypedStateSnapshot(
                module="InvarAST",
                ast_node="CallExpression",
                scope=safe_path,
                attributes=[
                    ("host", host_val),
                    ("method", method),
                    ("params", params_str),
                    ("code", code_slice[:250]),
                ],
            )
            scores = [
                ScoreQuestion(id=f"{endpoint_id}_impact", query=f"评估接口状态变更破坏力: {method} {safe_path[:60]}", active_levels=5),
                ScoreQuestion(id=f"{endpoint_id}_sensitivity", query=f"评估接口资产敏感度: {method} {safe_path[:60]}", active_levels=5),
            ]
            request = BatchDecisionRequest(state=state, choices=[], booleans=[], scores=scores)
            packed = packer.pack_request(request)

            seq_len = packed.input_ids.shape[1]

            # 第二道防线：Token 序列长度上限熔断保护 (严防 O(L^2) 显存爆炸)
            if seq_len > 2048:
                abnormal_samples_count += 1
                uniform = [0.2, 0.2, 0.2, 0.2, 0.2]
                results.append({
                    "endpoint_id": str(endpoint_id),
                    "impact_score": 1.0,
                    "sensitivity_score": 1.0,
                    "impact_probs": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "sensitivity_probs": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "latency_ms": 0.0,
                    "model_version": "qwen3-0.6b-dual-head-invar-v1",
                    "runner_provenance": "oversize_token_fallback",
                })
                continue

            input_ids = packed.input_ids.to(device)
            attention_mask = packed.attention_mask.to(device)

            # 第三道防线：CUDA OOM 显存熔断保护
            try:
                hidden_states = model.extract_hidden_states(input_ids, attention_mask)
                seq_hidden = hidden_states.squeeze(0)

                # Impact
                s_imp = packed.score_slices[0]
                q_imp = seq_hidden[s_imp.query_token_idx : s_imp.query_token_idx + 1]
                _, imp_dist_t, imp_exp_t = model.heads.forward_impact(q_imp)
                imp_dist = [round(float(p), 4) for p in imp_dist_t[0].cpu().numpy()]
                imp_exp = round(float(imp_exp_t[0][0].cpu().numpy()), 4)

                # Sensitivity
                s_sens = packed.score_slices[1]
                q_sens = seq_hidden[s_sens.query_token_idx : s_sens.query_token_idx + 1]
                _, sens_dist_t, sens_exp_t = model.heads.forward_sensitivity(q_sens)
                sens_dist = [round(float(p), 4) for p in sens_dist_t[0].cpu().numpy()]
                sens_exp = round(float(sens_exp_t[0][0].cpu().numpy()), 4)

                sample_elapsed_ms = (time.perf_counter() - t_sample_start) * 1000.0

                results.append({
                    "endpoint_id": str(endpoint_id),
                    "impact_score": imp_exp,
                    "sensitivity_score": sens_exp,
                    "impact_probs": imp_dist,
                    "sensitivity_probs": sens_dist,
                    "latency_ms": round(sample_elapsed_ms, 2),
                    "model_version": "qwen3-0.6b-dual-head-invar-v1",
                    "runner_provenance": "base-jev-system-one",
                })

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                abnormal_samples_count += 1
                results.append({
                    "endpoint_id": str(endpoint_id),
                    "impact_score": 1.0,
                    "sensitivity_score": 1.0,
                    "impact_probs": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "sensitivity_probs": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "latency_ms": 0.0,
                    "model_version": "qwen3-0.6b-dual-head-invar-v1",
                    "runner_provenance": "cuda_oom_fallback",
                })

            if idx % 100 == 0 or idx == total_endpoints:
                elapsed_so_far = time.perf_counter() - t_start_all
                rate = idx / elapsed_so_far if elapsed_so_far > 0 else 0
                print(f"  └─ 推理进度: [{idx}/{total_endpoints}] ({idx/total_endpoints*100:.1f}%) | 速度: {rate:.1f} 端点/秒", end="\r")

    print()
    t_total = time.perf_counter() - t_start_all

    with output_p.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 75)
    print(f" 🎉 base-jev 神经预测全量生成完毕！")
    print(f" 📄 输出预测产物: {output_p}")
    print(f" 🛡️  拦截超长异常: {abnormal_samples_count} 个")
    print(f" ⏱️  全量总耗时    : {t_total:.2f} 秒 (平均单端点全流程耗时: {t_total/max(1, total_endpoints)*1000.0:.2f} ms)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()


