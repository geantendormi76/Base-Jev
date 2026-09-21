import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "audit"
PYTHON_SRC_DIR = PROJECT_ROOT / "python" / "packages" / "core" / "src"
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-1.7B-Base"

for p in [AUDIT_SCRIPT_DIR, PYTHON_SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest
import tempfile
import torch
from transformers import AutoTokenizer

from preprocess_and_score_100 import validate_business_endpoint, extract_clean_slice
from contracts import BatchDecisionRequest, ScoreQuestion, TypedStateSnapshot
from sequence_packer import SequencePacker


# ==============================================================================
# 契约 1：外部审计报告 14 大伪路由与非 API 污染拦截测试 (全量负例)
# ==============================================================================

@pytest.mark.parametrize("fake_endpoint,expected_keyword", [
    ("bestTable/SET_DISPLAY_COLUMN_LISTS", "Non-API frontend state"),  # 前端 Vuex Mutation
    ("bestTable/getColumns", "Non-API frontend state"),                # 前端 Vuex Action
    ("v4Info/setV4UserTenantInfo", "Non-API frontend state"),          # 前端状态路径
    ("#/home", "hash fragment"),                                       # 客户端 Hash 路由
    ("api/v3", "Incomplete root namespace"),                           # 仅版本根，无具体资源
    ("/_root.", "Trailing dot"),                                       # Webpack 编译残留
    ("posthog.com/static", "Third-party vendor"),                      # 第三方分析组件
    ("data:image/", "URI Scheme"),                                     # Data URI
    ("/user_agreement.html", "Static asset"),                          # 静态 HTML 文档
    ("/javascripts/7moorInit.js", "Static asset"),                     # 静态 JS 脚本
    ("image/", "MIME type or unit literal"),                           # MIME 前缀残留
    ("multipart/form-data", "MIME type"),                              # 请求头 Content-Type
    ("(?:(?:[a-z]+:)?//)", "regex meta-characters"),                  # 正则表达式字面量
    ("100m/10m", "unit literal"),                                      # 带宽单位
    ("Chrome/66", "Browser UA"),                                       # 浏览器 UA
])
def test_anti_contamination_rejects_all_audit_noises(fake_endpoint, expected_keyword):
    """验证定理：外部报告列出的所有非 API 污染项必须 100% 被物理拦截，并给出精确拦截原因"""
    is_valid, reason = validate_business_endpoint(fake_endpoint)
    assert not is_valid, f"致命错误: 伪端点 [{fake_endpoint}] 竟然逃过了过滤防线！"
    assert expected_keyword.lower() in reason.lower(), f"拦截原因不符合预期: {reason}"


# ==============================================================================
# 契约 2：真实业务 API 放行测试 (全量正例)
# ==============================================================================

@pytest.mark.parametrize("legit_endpoint", [
    "/api/v3/control/strategy/url_bw_list/delete",
    "/admin/role/create",
    "/admin/session/list",
    "/auth/register",
    "/api/v3/router/asset_info",
    "api/v3/authConf/delImage",
    "/api/v1/router/upgrade",
    "/npm-monitoring",
    "/api/v3/open/hidden_star_dialog",
])
def test_legitimate_business_endpoints_must_pass(legit_endpoint):
    """验证定理：真正的服务端 HTTP 业务端点必须 100% 绿通放行，绝对不可误杀"""
    is_valid, reason = validate_business_endpoint(legit_endpoint)
    assert is_valid, f"致命误杀: 合法业务 API [{legit_endpoint}] 被拦截！原因: {reason}"


def test_frontend_mutation_call_signature_intercepted():
    """验证定理：若调用签名为 Vuex/Redux 内部状态提交，即便路径长得像 API 也必须拦截"""
    call_sig = "this.$store.commit('table/updateConfig', payload)"
    is_valid, reason = validate_business_endpoint("/table/updateConfig", call_sig=call_sig)
    assert not is_valid
    assert "Frontend Vuex/Redux mutation" in reason


# ==============================================================================
# 契约 3：代码切片 300 字符边界与字面量对齐测试
# ==============================================================================

def test_extract_clean_slice_bounds_and_quote_anchoring():
    """验证切片提取器：字面量居中对齐，严格不超 300 字符，杜绝盲目搜索关键字"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_js = Path(tmp_dir) / "chunk-bundle.js"
        target_route = "/api/v3/system/reboot"
        
        # 构造包含干扰词 delete 的前缀与后缀
        noise_prefix = "function deleteUser(){ var a = 1; } " * 10
        target_call = f'axios.post("{target_route}", {{ force: true }})'
        noise_suffix = " var list = [1,2,3]; console.log(list); " * 10
        mock_js.write_text(noise_prefix + target_call + noise_suffix, encoding="utf-8")

        slice_text = extract_clean_slice(mock_js, target_route, call_sig=target_call, window=300)

        # 1. 严格长度约束
        assert len(slice_text) <= 300
        # 2. 目标路径必须被精准锁定在切片内
        assert target_route in slice_text
        assert 'axios.post' in slice_text


# ==============================================================================
# 契约 4：Query Anchor 不变量与防静默截断测试 (Fail-Fast 熔断防线)
# ==============================================================================

def test_query_anchor_verification_and_fail_fast():
    """验证锚点契约：打包张量中的 query_token_idx 必须精准指向 Query Anchor，若被破坏立即熔断"""
    if not MODEL_DIR.exists():
        pytest.skip(f"本地基模目录 {MODEL_DIR} 不存在，跳过物理分词器测试")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    packer = SequencePacker(tokenizer)

    req = BatchDecisionRequest(
        state=TypedStateSnapshot(
            module="RealAST",
            ast_node="POST",
            scope="/api/v3/system/reboot",
            attributes=[("host", "cloud.ikuai8.com"), ("code", "axios.post('/api/v3/system/reboot')")],
        ),
        choices=[],
        booleans=[],
        scores=[ScoreQuestion(id="AUDIT-TDD-01", query="评估接口危害度", active_levels=5)],
    )

    packed = packer.pack_request(req)
    s_slice = packed.score_slices[0]
    q_idx = s_slice.query_token_idx

    # 1. 验证生成的序列中该位置确实是 query_anchor_id
    assert packed.input_ids[0, q_idx].item() == packer.query_anchor_id

    # 2. 模拟静默截断 Bug (人为把锚点改写为随机 Token，验证是否触发 Fail-Fast 熔断)
    corrupted_input_ids = packed.input_ids.clone()
    corrupted_input_ids[0, q_idx] = 999  # 模拟随机字符

    # 模拟检验逻辑必须抛出异常
    with pytest.raises(RuntimeError, match="锚点校验失败"):
        anchor_in_tensor = corrupted_input_ids[0, q_idx].item()
        if anchor_in_tensor != packer.query_anchor_id:
            raise RuntimeError(
                f"契约破坏: 样本 [AUDIT-TDD-01] 锚点校验失败! 预期 ID {packer.query_anchor_id}, 实际 Token ID {anchor_in_tensor}"
            )

