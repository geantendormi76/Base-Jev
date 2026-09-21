# AI Engineering Monorepo: Repository Closure Report
# Status: CLOSED
# Standard Reference: Mother Prompt Specification S8

## 1. Final Architecture (最终架构)
本项目已全面收敛至 AI Engineering Monorepo 工业级规范：
- `crates/core/`: 纯 Rust DirectML TensorCore 高吞吐推理引擎与出厂硬件质检 CLI。
- `python/packages/core/`: System-1 纯判别式认知反射算法底座（剥离自回归生成头，因果偏序 SequencePacker，认识论 Soft-CE + Brier + Entropy 联合损失）。
- `python/scripts/`: 核心模型训练、静态图导出与推理预测生产脚本。
- `python/tests/`: 37 项高覆盖率 Python 单元与契约测试集。
- `configs/profiles/`: 声明式训练与导出超参配方（`recipe_invar_0.6b.json`）。
- `scripts/`: 分类收敛为 `audit/`（取证审计）、`data/`（数据下载与提纯）、`build/`（权重单文件熔铸）。
- `docs/`: 标准化分流为 `ai/`（AI 唯一事实源与交接契约）、`architecture/`、`research/`、`decisions/`。

## 2. File Decisions Summary (文件裁决总结)
全量裁决详见 `docs/ai/FILE_DECISIONS.md`：
- MOVE: 38 项文件迁入规范 Monorepo 边界。
- KEEP: 20 项审计与脚手架工具原地保持。
- DELETE: 历史编译碎片 `src-tauri/target` 及遗留空外壳已物理清除。

## 3. Model Registry & Lifecycles (模型生命周期)
- 骨干底座：`models/Qwen3-0.6B-Base`（1024 维隐层特征）。
- 训练产物：`models/Qwen3-0.6B-Base-SystemOne-Invar/`（包含物理双头适配权重与真实评测指标）。
- 交付静态图：`models/system_one_unified.onnx`（1.25 GB FP16 单文件，带出厂护照 `MODEL_PASSPORT.json`）。

## 4. Data Lifecycle (数据生命周期)
- 遵循 `inbox/ -> raw/ -> interim/ -> processed/ -> fixtures/ -> manifests/` 标准分层。
- 零污染红线铁律维持：$Train \cap Gold = \emptyset$（物理零交叉验证通过）。

## 5. Path Migration & Ghost Paths Audit (幽灵路径清零)
- 详见 `docs/ai/PATH_MIGRATION.md`。
- 经过 S5 深度审计，源码与测试中对 `src-tauri/python/src`、`recipes/` 的历史硬编码幽灵路径残留数为 0。

## 6. Verification Evidence (双栈全量验证)
- Rust Core (`cargo test --workspace`): 3/3 PASSED (100% 绿通)。
- Python Core (`uv run pytest`): 37/37 PASSED, 4 SKIPPED (100% 绿通)。

## 7. Known Issues & Invariants (已知事实与不变量)
- I-01: Train ∩ Gold = ∅（黄金集零污染）。
- I-02: 预测端点逐行 1:1 强对齐。
- I-03: 纯张量输出，禁止自回归生成。
- I-04: 拒绝静默 3.0 兜底，Fail-Closed。

## 8. Current Safe Next Action (下一步安全动作)
- 进入 Phase 4: 双师共识融合与消融实验（Teacher Agreement Gate / Ensemble），为下游 Invar System-2 提供黄金综合训练集。




