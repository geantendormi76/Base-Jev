import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "python" / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import math
import random
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

# 尝试装配 Tree-sitter JavaScript AST 引擎 (对齐 ifm-ai/PRism-curator 黄金标准)
try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript as tsjavascript
    _HAS_TREE_SITTER = True
    _JS_LANGUAGE = Language(tsjavascript.language())
except Exception:
    _HAS_TREE_SITTER = False
    _JS_LANGUAGE = None

REPORT_PATH = Path("C:/dev/Invar/tmp/ikuai8_endpoints_report.json")
RAW_JS_DIR = Path("C:/dev/Invar/tmp/raw_js")
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-0.6B-Base"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "Qwen3-0.6B-Base-SystemOne-Invar"

PREPROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "invar_preprocessed_100_real.jsonl"
SCORED_RESULT_PATH = PROJECT_ROOT / "data" / "invar_preprocessed_100_scored_dual.jsonl"

SCHEME_RE = re.compile(r"^(?:data|javascript|mailto|blob|file|about|tel):", re.IGNORECASE)
MIME_TYPE_RE = re.compile(r"^(?:text|application|image|audio|video|font|multipart)/", re.IGNORECASE)
STATIC_EXT_RE = re.compile(r"\.(?:html?|jsx?|tsx?|css|less|scss|map|png|jpe?g|gif|svg|ico|woff2?|ttf|eot|mp[34]|wasm|txt)$", re.IGNORECASE)
REGEX_META_RE = re.compile(r"[\\^$*+?()\[\]{}|]")
CLIENT_HASH_RE = re.compile(r"#")
BANDWIDTH_UNIT_RE = re.compile(r"^\d+(?:\.\d+)?[gmkb]/\d+(?:\.\d+)?[gmkb]?$", re.IGNORECASE)
KNOWN_UNIT_LITERALS = {"100m/10m", "2.5g/1g", "kb/s", "mb/s", "gb/s", "data", "[getter/setter]", "image/"}
BROWSER_UA_RE = re.compile(r"^(?:chrome|firefox|safari|opera)/\d+", re.IGNORECASE)
CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]")
THIRD_PARTY_RE = re.compile(r"redux|react|vue|webpack|babel|core-js|lodash|moment|posthog", re.IGNORECASE)
LEGIT_API_PREFIX_RE = re.compile(r"^(?:api/|v\d+/|rest/|rpc/|service/)", re.IGNORECASE)
CONSECUTIVE_SLASH_RE = re.compile(r"/{2,}")


class TreeSitterAstExtractor:
    """
    对齐 ifm-ai/PRism-curator 黄金标准的抽象语法树提取器
    通过 CallExpression 与封闭函数作用域绑定，从根源切断相邻接口跨语句泄漏
    """
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

    def _parse_tree(self, physical_file: Path, source_bytes: bytes) -> Any:
        file_key = str(physical_file.resolve())
        if file_key in self._tree_cache:
            return self._tree_cache[file_key]
        tree = self.parser.parse(source_bytes)
        self._tree_cache[file_key] = tree
        return tree

    def find_call_expression(self, tree: Any, path_str: str) -> Optional[Tuple[Any, str, int, int]]:
        if tree is None:
            return None

        stack = [tree.root_node]
        while stack:
            curr = stack.pop()
            if curr.type == "call_expression":
                args_node = curr.child_by_field_name("arguments")
                if not args_node:
                    for ch in curr.children:
                        if ch.type == "arguments":
                            args_node = ch
                            break

                if args_node:
                    for arg in args_node.named_children:
                        arg_text = arg.text.decode("utf-8", errors="ignore")
                        if path_str in arg_text:
                            call_sig = curr.text.decode("utf-8", errors="ignore")
                            return curr, call_sig, curr.start_byte, curr.end_byte

                        if arg.type == "object":
                            for prop in arg.named_children:
                                if prop.type in ("pair", "property"):
                                    prop_text = prop.text.decode("utf-8", errors="ignore")
                                    if path_str in prop_text:
                                        call_sig = curr.text.decode("utf-8", errors="ignore")
                                        return curr, call_sig, curr.start_byte, curr.end_byte

            for child in reversed(curr.named_children):
                stack.append(child)
        return None

    def extract_slice(self, file_p: Path, path_str: str, call_sig: str) -> Tuple[str, str]:
        """
        返回: (clean_slice, extraction_source)
        """
        if not file_p.is_file():
            fallback = call_sig if (call_sig and len(call_sig) >= 6) else path_str
            return fallback[:self.window_chars].strip(), "MissingFileFallback"

        try:
            content = file_p.read_text(encoding="utf-8", errors="ignore")
            source_bytes = content.encode("utf-8", errors="ignore")
        except Exception:
            return path_str[:self.window_chars], "ReadErrorFallback"

        # 1. 优先使用 Tree-sitter AST 语法树调用绑定
        if self.parser is not None:
            try:
                tree = self._parse_tree(file_p, source_bytes)
                ast_match = self.find_call_expression(tree, path_str)
                if ast_match is not None:
                    call_node, _, start_byte, end_byte = ast_match

                    # 向上绑定至封闭函数或方法定义域 (Function Scope Binding)
                    enclosing = call_node
                    parent = call_node.parent
                    while parent is not None and parent.type != "program":
                        if parent.type in (
                            "function_declaration",
                            "function_expression",
                            "arrow_function",
                            "method_definition",
                            "export_statement",
                        ):
                            enclosing = parent
                            break
                        parent = parent.parent

                    enclosing_text = enclosing.text.decode("utf-8", errors="ignore")
                    if len(enclosing_text) <= self.window_chars:
                        clean_slice = re.sub(r"\s+", " ", enclosing_text).strip()
                    else:
                        target_len = end_byte - start_byte
                        surplus = max(0, self.window_chars - target_len)
                        half_left = surplus // 2
                        half_right = surplus - half_left

                        s = max(0, start_byte - half_left)
                        e = min(len(source_bytes), end_byte + half_right)
                        raw_text = source_bytes[s:e].decode("utf-8", errors="ignore")
                        clean_slice = re.sub(r"\s+", " ", raw_text).strip()[:self.window_chars]

                    return clean_slice, "TreeSitter_AST_CallExpression"
            except Exception:
                pass

        # 2. 保底：严格带引号字面量边界对齐
        idx = -1
        target_len = len(path_str)
        for quote in ['"', "'", "`"]:
            quoted_target = f"{quote}{path_str}{quote}"
            found_idx = content.find(quoted_target)
            if found_idx != -1:
                idx = found_idx
                target_len = len(quoted_target)
                break

        if idx == -1 and call_sig and len(call_sig) >= 8:
            found_idx = content.find(call_sig)
            if found_idx != -1:
                idx = found_idx
                target_len = len(call_sig)

        if idx == -1:
            idx = content.find(path_str)
            target_len = len(path_str)

        if idx != -1:
            half_left = min(80, self.window_chars // 3)
            half_right = min(120, self.window_chars - half_left - target_len)
            start = max(0, idx - half_left)
            end = min(len(content), idx + target_len + half_right)
            raw_slice = content[start:end]
            return re.sub(r"\s+", " ", raw_slice).strip()[:self.window_chars], "ExactLiteralFallback"

        fallback = call_sig if (call_sig and len(call_sig) >= 6) else path_str
        return f"// [Endpoint Context] {fallback[:self.window_chars]}".strip(), "LiteralFallback"


def validate_business_endpoint(path_str: str, call_sig: str = "") -> Tuple[bool, str]:
    if not path_str or len(path_str) < 3:
        return False, "Path too short (<3 chars)"

    p = path_str.strip()

    if REGEX_META_RE.search(p):
        return False, "Contains regex meta-characters"
    if CONSECUTIVE_SLASH_RE.search(p):
        return False, "Contains consecutive redundant slashes (e.g. ///)"

    if SCHEME_RE.search(p):
        return False, "Matches URI Scheme"

    if CLIENT_HASH_RE.search(p):
        return False, "Contains client-side hash fragment (#)"


    p_clean = p.split("?")[0].split("#")[0]
    if STATIC_EXT_RE.search(p_clean):
        return False, "Static asset/file extension"

    p_lower = p.lower()
    p_stripped = p_lower.strip("/")
    if MIME_TYPE_RE.match(p_stripped) or p_stripped in KNOWN_UNIT_LITERALS or p_lower in KNOWN_UNIT_LITERALS:
        return False, "MIME type or unit literal"

    if BANDWIDTH_UNIT_RE.match(p_stripped):
        return False, "Bandwidth speed literal"

    if BROWSER_UA_RE.match(p_stripped):
        return False, "Browser UA version string"

    if CHINESE_RE.search(p) or "\n" in p or "\r" in p:
        return False, "Contains Chinese or newline characters"

    if THIRD_PARTY_RE.search(p):
        return False, "Third-party vendor/library pattern"

    if not p.startswith("/") and not LEGIT_API_PREFIX_RE.match(p):
        return False, "Non-API frontend state/action path (missing leading / or api/ prefix)"

    segments = [s for s in p_stripped.split("/") if s]
    if len(segments) == 0:
        return False, "Empty path / no valid resource segments"

    if len(segments) < 2 and not p.startswith("/"):
        return False, "Incomplete path namespace"

    if len(segments) == 2 and segments[0].lower() == "api" and re.match(r"^v\d+$", segments[1].lower()):
        return False, "Incomplete root namespace (e.g. api/v3 without resource)"

    if p.endswith("."):
        return False, "Trailing dot compilation artifact"

    if not all(re.match(r"^[A-Za-z0-9_.~:?@!&',;=%-]+$", s) for s in segments):
        return False, "Contains segment with illegal characters"

    if call_sig and re.search(r"\b(?:commit|dispatch|\$store|\$emit|setState)\b", call_sig, re.IGNORECASE):
        return False, "Frontend Vuex/Redux mutation call signature"

    return True, "Valid API Endpoint"


def main():
    print("=" * 75)
    print(" 🚀 base-jev: 0.6B 模型【Tree-sitter AST 精确切片与硬契约双正交终审】")
    print(f" 📂 骨干底座 : {BASE_MODEL_DIR.name}")
    print(f" 📂 适配权重 : {CHECKPOINT_DIR.name}")
    print(f" 🌳 AST 引擎 : {'✅ Tree-sitter JavaScript 就绪' if _HAS_TREE_SITTER else '⚠️ Tree-sitter 缺失，使用保底'}")
    print("=" * 75)

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"找不到 Invar 报告: {REPORT_PATH}")

    raw_data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    raw_endpoints = raw_data.get("endpoints", [])

    # 1. 严格排除训练集与评估集已消耗过的端点 (真·盲测)
    train_val_keys = set()
    for dataset_name in ["distilled_invar_train_dual.jsonl", "distilled_invar_eval_dual.jsonl", "distilled_invar_train.jsonl", "distilled_invar_eval.jsonl"]:
        dataset_file = PROJECT_ROOT / "data" / dataset_name
        if dataset_file.is_file():
            with dataset_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            m = str(row.get("method", "GET")).upper()
                            p = str(row.get("path", "")).strip()
                            if p:
                                train_val_keys.add(f"{m}|{p}")
                        except Exception:
                            pass

    clean_candidates = []
    seen_dedup = set()

    for ep in raw_endpoints:
        path = str(ep.get("path", "")).strip()
        method = str(ep.get("method", "GET")).upper()
        call_sig = str(ep.get("call_signature", ""))

        is_valid, _ = validate_business_endpoint(path, call_sig)
        if not is_valid:
            continue

        dedup_key = f"{method}|{path}"
        if dedup_key in seen_dedup:
            continue
        seen_dedup.add(dedup_key)

        if dedup_key in train_val_keys:
            continue

        clean_candidates.append(ep)

    actual_count = len(clean_candidates)
    print(f"  ├─ 排除已见端点并切除 /// 漏洞后，纯净业务端点实有: {actual_count} 个")

    if actual_count < 100:
        sampled_eps = clean_candidates
    else:
        rng = random.Random(2026)
        sampled_eps = rng.sample(clean_candidates, 100)

    # 2. 使用 Tree-sitter 提取精准无泄漏代码切片
    ast_extractor = TreeSitterAstExtractor(window_chars=250)
    preprocessed_samples = []
    ast_sources_counter = Counter()

    for i, ep in enumerate(sampled_eps, 1):
        source_file = ep.get("source_file", "")
        method = str(ep.get("method", "GET")).upper()
        path = str(ep.get("path", "")).strip()
        call_sig = str(ep.get("call_signature", ""))

        parts = Path(source_file).parts
        host = parts[1] if len(parts) > 1 and parts[0] == "raw_js" else "cloud.ikuai8.com"
        file_name = Path(source_file).name
        target_js = RAW_JS_DIR / host / file_name
        if not target_js.is_file():
            found = list(RAW_JS_DIR.rglob(file_name))
            target_js = found[0] if found else target_js

        code_slice, slice_source = ast_extractor.extract_slice(target_js, path, call_sig)
        ast_sources_counter[slice_source] += 1
        clean_params = [str(p)[:30] for p in ep.get("extracted_params", [])][:8]

        sample_item = {
            "test_id": f"AUDIT-{i:03d}",
            "host": host,
            "method": method,
            "path": path,
            "params": clean_params,
            "tags": ep.get("tags", [])[:5],
            "code_slice": code_slice,
            "slice_source": slice_source,
            "query": f"评估接口 [{method} {path}] 的安全属性",
        }
        preprocessed_samples.append(sample_item)

    print(f"  └─ 切片提取拓扑统计: {dict(ast_sources_counter)}")

    # 3. 装载 0.6B 双正交模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("必须使用 NVIDIA GPU 执行推理！")

    print(f"\n[*] 正在装载 0.6B 双正交决策模型至 RTX 3060 ({CHECKPOINT_DIR.name})...")
    torch.cuda.empty_cache()

    model = SystemOneModel.from_pretrained_base(BASE_MODEL_DIR, torch_dtype=torch.float16)
    model.backbone = PeftModel.from_pretrained(model.backbone, str(CHECKPOINT_DIR / "backbone_lora"))
    model.heads.load_state_dict(torch.load(CHECKPOINT_DIR / "decision_heads.pt", map_location=device))
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), trust_remote_code=True)
    packer = SequencePacker(tokenizer)

    print(f"[*] 启动单前向双正交决策推演 (带强断言保护)...")
    t0 = time.perf_counter()

    scored_records = []
    cartesian_matrix = np.zeros((5, 5), dtype=int)
    token_lengths = []
    impact_confidences = []
    sens_confidences = []

    with torch.inference_mode():
        for item in preprocessed_samples:
            state = TypedStateSnapshot(
                module="RealAST",
                ast_node="CallExpression",
                scope=item["path"][:80],
                attributes=[
                    ("host", item["host"]),
                    ("method", item["method"]),
                    ("params", ",".join(item["params"])),
                    ("code", item["code_slice"]),
                ],
            )
            
            # 同时挂载两道正交问题锚点
            scores = [
                ScoreQuestion(
                    id=f"{item['test_id']}_impact",
                    query=f"评估该接口状态变更与底层破坏力 (1~5): {item['query']}",
                    active_levels=5,
                ),
                ScoreQuestion(
                    id=f"{item['test_id']}_sensitivity",
                    query=f"评估该接口交互资产机密性与敏感度 (1~5): {item['query']}",
                    active_levels=5,
                ),
            ]
            request = BatchDecisionRequest(state=state, choices=[], booleans=[], scores=scores)
            packed = packer.pack_request(request)

            seq_len = packed.input_ids.shape[1]
            token_lengths.append(seq_len)

            # ==================================================================
            # 🚨 强契约断言防线 (恢复 Audit 扣分项，Fail-Fast 熔断)
            # ==================================================================
            if seq_len > 2048:
                raise RuntimeError(f"契约破坏: 样本 [{item['test_id']}] 序列长度 ({seq_len}) 超过上限 2048！")

            if len(packed.score_slices) < 2:
                raise RuntimeError(f"契约破坏: 样本 [{item['test_id']}] 缺少双正交切片！")

            s_imp = packed.score_slices[0]
            s_sens = packed.score_slices[1]

            # 严格断言输入张量中的对应下标确实为原子锚点 Token
            imp_token_id = packed.input_ids[0, s_imp.query_token_idx].item()
            sens_token_id = packed.input_ids[0, s_sens.query_token_idx].item()

            if imp_token_id != packer.query_anchor_id:
                raise RuntimeError(
                    f"契约破坏: 样本 [{item['test_id']}] Impact 锚点无效! 预期 {packer.query_anchor_id}, 实得 {imp_token_id}"
                )
            if sens_token_id != packer.query_anchor_id:
                raise RuntimeError(
                    f"契约破坏: 样本 [{item['test_id']}] Sensitivity 锚点无效! 预期 {packer.query_anchor_id}, 实得 {sens_token_id}"
                )

            input_ids = packed.input_ids.to(device)
            attention_mask = packed.attention_mask.to(device)

            hidden_states = model.extract_hidden_states(input_ids, attention_mask)
            seq_hidden = hidden_states.squeeze(0)

            # 1. 结算破坏力维度 (Impact)
            q_imp_vec = seq_hidden[s_imp.query_token_idx : s_imp.query_token_idx + 1]
            _, imp_dist_t, imp_exp_t = model.heads.forward_score(q_imp_vec)
            imp_dist = [round(float(p), 4) for p in imp_dist_t[0].cpu().numpy()]
            imp_exp = round(float(imp_exp_t[0][0].cpu().numpy()), 2)
            imp_dom = int(np.argmax(imp_dist) + 1)
            imp_conf = float(np.max(imp_dist))

            # 2. 结算敏感度维度 (Sensitivity)
            q_sens_vec = seq_hidden[s_sens.query_token_idx : s_sens.query_token_idx + 1]
            _, sens_dist_t, sens_exp_t = model.heads.forward_score(q_sens_vec)
            sens_dist = [round(float(p), 4) for p in sens_dist_t[0].cpu().numpy()]
            sens_exp = round(float(sens_exp_t[0][0].cpu().numpy()), 2)
            sens_dom = int(np.argmax(sens_dist) + 1)
            sens_conf = float(np.max(sens_dist))

            # 代数一致性断言
            if not math.isclose(sum(imp_dist), 1.0, abs_tol=1e-3):
                raise RuntimeError(f"代数破坏: Impact 概率和不归一 ({sum(imp_dist)})")
            if not math.isclose(sum(sens_dist), 1.0, abs_tol=1e-3):
                raise RuntimeError(f"代数破坏: Sensitivity 概率和不归一 ({sum(sens_dist)})")

            cartesian_matrix[imp_dom - 1, sens_dom - 1] += 1
            impact_confidences.append(imp_conf)
            sens_confidences.append(sens_conf)

            scored_records.append({
                "test_id": item["test_id"],
                "route": f"{item['method']} {item['path']}",
                "params": item["params"],
                "tags": item["tags"],
                "ast_node_type": "CallExpression",
                "slice_source": item["slice_source"],
                "query_anchor_verified": True,
                "seq_token_length": seq_len,
                "impact": {
                    "predicted_score": imp_exp,
                    "dominant_level": f"Level {imp_dom}",
                    "confidence": round(imp_conf, 4),
                    "distribution": imp_dist,
                },
                "sensitivity": {
                    "predicted_score": sens_exp,
                    "dominant_level": f"Level {sens_dom}",
                    "confidence": round(sens_conf, 4),
                    "distribution": sens_dist,
                },
                "code_snippet": item["code_slice"][:120] + "...",
            })

    total_time = time.perf_counter() - t0
    avg_latency = (total_time / max(1, len(preprocessed_samples))) * 1000.0

    with SCORED_RESULT_PATH.open("w", encoding="utf-8") as f:
        for r in scored_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 75)
    print(f" 🎉 双正交二维终审推演竣工 (实测 {len(preprocessed_samples)} 条纯净未见端点)！")
    print(f" ⏱️  推演总耗时         : {total_time:.2f} 秒 (平均单前向双正交延迟: {avg_latency:.2f} ms)")
    print(f" 🛡️  硬契约校验         : ✅ 100% 样本满足 seq_len <= 2048 且双 Query Anchor 物理对齐！")
    print(f" 🎯 破坏力平均置信度   : {sum(impact_confidences)/len(impact_confidences)*100:.1f}%")
    print(f" 🎯 敏感度平均置信度   : {sum(sens_confidences)/len(sens_confidences)*100:.1f}%")
    print(f" 📦 最终双正交结果库   : {SCORED_RESULT_PATH.resolve()}")
    print("=" * 75)

    print("\n[📊 二维笛卡尔分布矩阵 (5x5 Grid: 纵轴=破坏力 Impact | 横轴=敏感度 Sensitivity)]:")
    print("               Sens-L1   Sens-L2   Sens-L3   Sens-L4   Sens-L5")
    for i in range(5):
        row_str = f"  Impact-L{i+1}: "
        for j in range(5):
            row_str += f"{cartesian_matrix[i, j]:8d}"
        print(row_str)

    print("\n[🎯 关键争议样本双正交解耦实况]:")
    for s in scored_records:
        r = s["route"]
        if any(target in r for target in ["/admin/role/get", "/alarm_message/export", "/admin/session/evict", "/permission", "/auth/register"]):
            print(f"  • {r:42s} (切片来源: {s['slice_source']})")
            print(f"    ├─ 破坏力 (Impact)     : {s['impact']['predicted_score']:4.2f} ({s['impact']['dominant_level']}, 置信度: {s['impact']['confidence']:.2f})")
            print(f"    └─ 敏感度 (Sensitivity): {s['sensitivity']['predicted_score']:4.2f} ({s['sensitivity']['dominant_level']}, 置信度: {s['sensitivity']['confidence']:.2f})")

if __name__ == "__main__":
    main()


def extract_clean_slice(file_p: Path, path_str: str, call_sig: str = "", window: int = 250) -> str:
    extractor = TreeSitterAstExtractor(window_chars=window)
    clean_slice, _ = extractor.extract_slice(Path(file_p), path_str, call_sig)
    return clean_slice

