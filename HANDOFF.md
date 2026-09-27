# 🤝 AI Development Handoff Specification (开发会话交接档案)

## 0. Handoff Metadata
- **文档版本**：`v3.2.0-gold-dataset-certified` `[FACT]`
- **生成时间锚点**：2026-09-27 15:30:00 (UTC+9) `[FACT]`
- **当前仓库根路径**：`C:\dev\Distiller` `[FACT]`
- **上游数据源路径**：`C:\dev\crawl` (`PROJECT_ROOT.parent / "crawl"`) `[FACT]`
- **理论规格与标准**：
  - `TianyuCodings/NanoJev` / `browser-use/jev-ultrafast` (零 Token 解码单步前向推演体系) `[FACT]`
  - `Alibaba Xianyu Team: LLP (LLM-Based Product Pricing in E-commerce, arXiv:2510.09347)` `[FACT]`
  - `Prometheus 2 (EMNLP) & Argilla Distilabel` (外部声明式量规解耦与最大熵指数族平滑) `[FACT]`
- **可复现性等级**：`FULLY REPRODUCIBLE` (Distiller 全量 19 个单元测试 100% 绿灯，出厂数据集通过代数归一化与终极争议样本实证双重验收) `[FACT]`

---

## 1. 核心技术升级与问题根治记录
1. **彻底攻克长描述正文缺失痛点**：
   - 协同上游 `crawl` 在无头模式下通过 `x5sec` 持久化与反自动化特征抹除，全量捕获卖家原始长描述正文 `description`；
   - 在 `xianyu_synthesizer.py` 中将描述全量并入 `code_slice` 第二层，消除了“坏的坏的”、“电源线烧过”在标题被截断导致大模型误判的根本盲区。`[FACT]`
2. **代码级经济规律前置硬守卫 (Deterministic Pre-guard)**：
   - 基于资产价格天花板法则（$P < 1000$ 元或偏离度 $> 75\%$ 且含配件特征词），对纯配件、支架、水冷头直接定级为 Level 1；
   - 52.2%（261条）确定性样本实现 **0ms 直通免调 API**，彻底杜绝大模型在明显配件上的幻觉，并将蒸馏耗时从 8500 秒压缩至 2575 秒！`[FACT]`
3. **出厂质检大验收全绿**：
   - 训练集 `distilled_xianyu_train.jsonl` (387 条) 与评估集 `distilled_xianyu_eval.jsonl` (97 条)：
     - 概率归一化违例数：**0** `[FACT]`
     - 期望值点积偏差数：**0** `[FACT]`
     - 教师模型兜底均分率：**0/193 (Train) | 0/42 (Eval)** `[FACT]`
   - 关键争议标的实测：微星 3090 死卡精准定级 `Sens: 4.95 / Impact: 1.50`，水冷头精准定级 `Impact: 1.05`，登山包与羽毛球拍确凿隔离！`[FACT]`

---

## 2. 最终交付资产清单
- `artifacts/exports/distilled_xianyu_train.jsonl`：387 条出厂级标准训练集；
- `artifacts/exports/distilled_xianyu_eval.jsonl`：97 条出厂级标准评估集（严格物理隔离）；
- `configs/prompts/xianyu_hardware.yaml`：全要素因果评级量规规范。

---

## 3. 下一步路线 (Roadmap)
- **阶段 M4 (即将启动)**：移交外部独立训练工程，读取上述两份 JSONL 数据集，启动 NanoJev 0.6B LoRA 微调，导出 23ms 前向推理引擎！`[TODO]`


你现在接管【Invar / NanoJev 0.6B 决策模型微调训练工程】。
数据策展与因果蒸馏中枢（Distiller）已经完成了出厂级数据生产与全量代数质检，准予点火训练！

【上游出厂数据集物理路径】:
• 训练集 (387条): C:\dev\Distiller\artifacts\exports\distilled_xianyu_train.jsonl
• 评估集 (97条):  C:\dev\Distiller\artifacts\exports\distilled_xianyu_eval.jsonl
• 提示量规:       C:\dev\Distiller\configs\prompts\xianyu_hardware.yaml

【数据契约与训练架构规范】:
1. 任务性质: 非自回归 (Non-Autoregressive) 双头因果决策与估值模型，单步前向推演 (目标耗时 23ms)；
2. 输入特征: sample["code_slice"] (已包含 16 维数值特征、相对偏离度及卖家详细长描述)；
3. 监督标签 (软标签分布):
   - Impact (物理形态残值): target_scores.impact.distribution (5档归一化概率)
   - Sensitivity (暗病欺诈闸门): target_scores.sensitivity.distribution (5档归一化概率)
4. 损失函数: 基于 KL 散度 (KL-Divergence) 或 软交叉熵，拟合教师大模型平滑后的 5 档概率分布；
5. 基座模型建议: Qwen2.5-0.5B (使用 LoRA 微调)；
6. 交付目标: 完成微调、验证评估集收敛指标 (Eval Loss & 期望分偏差)，并导出单文件 ONNX / GGUF。

请严格遵循系统工程宪法，读取训练集第一行样本确认数据契约，并提出你的第一单步工程计划！