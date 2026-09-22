# 📋 Base-Jev 核心待办事项与技术债务备忘录 (Backlog & Tech Debt)

> **任务标识**: `TASK-BASE-JEV-ONNX-DUAL-HEAD-V2`  
> **关联项目**: `C:\dev\Base-Jev` (模型工厂) $\leftrightarrow$ `C:\dev\Invar` (下游单兵执行引擎)  
> **记录时间**: 2026-09-22  
> **状态**: `PENDING / SEALED (保留事实，待后续升级)`  
> **优先级**: `P1 - 架构解耦与自包含发货重构`

---

## 1. 发现事实依据 (Factual Ground Truth)

在对 `Base-Jev` 训练导出的 `system_one_unified.onnx` 和推理流水线进行实地探查时，确认以下客观事实：

### 事实 1：物理计算图的输入输出张量现状
通过 ONNX 拓扑直读探查确认，现有 `system_one_unified.onnx`（1.28 GB, FP16）的拓扑仅包含以下节点：
* **输入张量 (Inputs)**:
  `input_ids`, `attention_mask`, `choice_q_pos`, `choice_cand_pos`, `choice_mask`, `bool_q_pos`, `score_q_pos`
* **输出张量 (Outputs)**:
  `choice_probs`, `bool_probs`, `score_dist`, `score_expected`

### 事实 2：PyTorch 训练态与 ONNX 导出态的严重错位 (Discrepancy)
1. **PyTorch 源码中的真实实现 (`decision_heads.py`)**：
   网络定义了**两个独立的物理打分头**，且在 `trainer.py` 中分别参与反向传播：
   ```python
   self.impact_head = ScoreOrdinalHead(hidden_size, score_max_levels)      # 破坏力打分头
   self.sensitivity_head = ScoreOrdinalHead(hidden_size, score_max_levels) # 敏感度打分头
   self.score_head = self.impact_head                                      # 旧单头兼容别名
   ```
2. **ONNX 导出包装类中的硬编码约束 (`onnx_export_wrapper.py`)**：
   在导出时，`SystemOneOnnxWrapper` 仅连通了单打分头：
   ```python
   score_q_vecs = torch.index_select(seq_hidden, 0, score_q_pos)
   _, score_dist, score_expected = self.heads.forward_score(score_q_vecs)
   # 注: forward_score() 在 decision_heads.py 中内部硬编码只调用 self.impact_head!
   ```
3. **`self.sensitivity_head`（敏感度头）的实际权重**：
   在静态图编译期间，由于没有连接到任何输出张量，敏感度头在 ONNX 导出时**被视为 Dead Code 剪枝，或者未被激活**。

### 事实 3：推演脚本被迫退回 PyTorch 模式的真正根因
在 `Base-Jev/python/scripts/generate_triage_predictions.py` 中，开发者之所以绕过了已导出的 ONNX 模型，转而使用重型 PyTorch 运行时推演，是因为：
```python
# 必须直接在 PyTorch 实例中分别调用双头，才能提取真实的双正交分数：
_, imp_dist_t, imp_exp_t = model.heads.forward_impact(q_imp)
_, sens_dist_t, sens_exp_t = model.heads.forward_sensitivity(q_sens)
```
**如果直接拿现有的 ONNX 计算图推演，第二维敏感度分数实际全是由破坏力权重计算出来的假数据。**

---

## 2. 产生的架构问题与技术债务 (Architecture Impact)

1. **违背单兵作战原则**：
   `Invar` 无法完全脱离 `Base-Jev`。每次有新靶场需要执行 System-1 接口意图预测时，下游必须强行依赖 `Base-Jev` 的重型 GPU 环境（PyTorch + Transformers + CUDA + PEFT）。
2. **出厂护照（Passport）定义脱节**：
   `MODEL_PASSPORT.json` 中声明了 `DualScore` 原语，但物理单文件 ONNX 计算图实际上仅导出了单头 `Score` 算子。

---

## 3. 长远系统升级方案 (Target Architecture & Roadmap)

未来系统升级时，需按照以下 4 步执行端到端物理闭环：

### 步骤 A：升级 ONNX 导出包装器 (`onnx_export_wrapper.py`)
重构 `SystemOneOnnxWrapper`，显式接入物理双头计算图，支持双正交打分张量并行输出：
```python
# 改造目标:
def forward(self, ..., imp_q_pos: torch.Tensor, sens_q_pos: torch.Tensor):
    ...
    # Impact 破坏力头
    imp_vecs = torch.index_select(seq_hidden, 0, imp_q_pos)
    _, imp_dist, imp_expected = self.heads.forward_impact(imp_vecs)
    
    # Sensitivity 敏感度头
    sens_vecs = torch.index_select(seq_hidden, 0, sens_q_pos)
    _, sens_dist, sens_expected = self.heads.forward_sensitivity(sens_vecs)
    
    return choice_probs, bool_probs, imp_dist, imp_expected, sens_dist, sens_expected
```

### 步骤 B：重新执行烧结融合与 ONNX 导出 (`export_to_onnx.py`)
1. 注册新的输出张量名称：
   * `impact_dist`, `impact_expected`
   * `sensitivity_dist`, `sensitivity_expected`
2. 动态轴绑定更新为 `(num_scores, 5)` 与 `(num_scores, 1)`；
3. 产出全新自包含二进制图：`system_one_dual_v2.onnx`。

### 步骤 C：更新 Rust 硬件质检台 (`crates/core/src/engine.rs` & `main.rs`)
1. 调整 DirectML 运行时的解构元组，读取 `impact_expected` 和 `sensitivity_expected`；
2. 运行 `cargo run --bin base-jev-qa`，重新验证 RTX 3060 DirectML 稳态延迟（目标依旧维持在 $\le 30\text{ms}$）；
3. 重新签发更新版 `MODEL_PASSPORT.json` (v2.0.0)。

### 步骤 D：在 Invar 仓内落地极简纯 ONNX 推理器
在 `Invar/scripts/data/pipeline/`（或 `python/scripts/`）下编写完全自包含的 `predict_triage_onnx.py`：
* 依赖仅包含：`onnxruntime-directml` + `tokenizers`（纯 Rust 绑定的 Fast Tokenizer）+ `numpy`；
* 零 PyTorch，零 CUDA Toolkit；
* Invar 用户直接运行单命令即可一键完成万级端点的 23ms 秒级推演，达成**彻底的物理单兵化交付**。

---

## 4. 当前运行基准物证备忘 (Current Verified Baseline)

* 当前用于对齐的推演物证：`C:\dev\Invar\tmp\base_jev_predictions_1216.jsonl`
* 该物证由 `Base-Jev` 在 GPU 真实双头（`forward_impact` + `forward_sensitivity`）全精度计算下产出，精度与特征分布完整且经过 1:1 对齐检验，可作为后续 ONNX v2 导出的**绝对对照黄金真值（Ground Truth Benchmark）**。

