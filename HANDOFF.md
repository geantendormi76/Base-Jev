# AI Development Handoff Specification (开发会话交接档案)

## 0. Handoff Metadata
- **文档版本**：`v4.1.0-invar-cuda-verified` `[FACT]`
- **生成时间锚点**：2026-10-02 22:45:00 (UTC+8) `[FACT]`
- **当前宿主操作系统**：Linux Mint 22.3 Cinnamon (x86_64, Kernel 6.8.0, 物理硬件宿主机) `[FACT]`
- **核心计算硬件**：NVIDIA GeForce RTX 3060 (12GB VRAM, CUDA Compute Capability 8.6) `[FACT]`
- **关键工程根目录**：
  - 数据生产中枢：`/home/zhz/Distiller` `[FACT]`
  - 模型制造工厂：`/home/zhz/Base-Jev` (当前工作目录) `[FACT]`
  - 端侧落地服务：`/home/zhz/Invar` `[FACT]`
- **文档唯一目标**：记录 Linux Mint 22.3 原生 CUDA 硬件加速打通、Rust 质检 7.85 ms 全量绿通、出厂护照正式签发的客观事实，支持后续业务无缝消费。`[DECISION]`

## 1. Project Identity
- **项目名称**：Base-Jev (System-1 神经认知决策模型工厂) `[FACT]`
- **工程定位**：面向边缘端非自回归 (Non-Autoregressive) 毫秒级决策反射模型的高性能微调、图编译压铸与纯 Rust 原生出厂硬件质检台。`[FACT]`
- **主技术栈**：
  - Python 域：`Python 3.12` + `uv` (`torch 2.6.0+cu124`, `peft`, `onnxruntime-gpu 1.20.2`) `[FACT]`
  - Rust 域：`Rust 2021` + `Cargo` (`ort 2.0.0-rc.13`, `tokenizers`, `clap`, `half`) `[FACT]`
  - 底座模型：`Qwen/Qwen3-0.6B-Base` (Hidden Size: 1024, 16 Heads, 8 KV-Heads) `[FACT]`

## 2. Current Mission
完成从 Windows 到 Linux Mint 22.3 迁移后 Invar 决策模型全链路重铸：
Distiller (v2 纯多正交数据) ➔ Base-Jev (微调 & ONNX 压铸) ➔ base-jev-qa (Rust 出厂质检) ➔ Invar (业务消费)
当前已在 Linux 原生 RTX 3060 硬件上跑通 `base-jev-qa`，实测稳态延迟 **7.85 ms**，突破 45.0 ms 红线，带数字印章的《模型出厂护照》`MODEL_PASSPORT.json` 已正式签发。`[FACT]`

## 3. Current Objective
**【当前状态：Phase 1 硬件质检已圆满竣工】**
待办推进点：将 `artifacts/packages/invar-intent-0.6b-v2/` 标准交付三件套（`system_one_unified.onnx`, `tokenizer.json`, `MODEL_PASSPORT.json`）无缝交付至下游服务工程 `/home/zhz/Invar/models/`，并进行业务端到端推理闭环测试。`[FACT]`

## 4. Last Known Good State
- **时间锚点**：2026-10-02 22:42:00 (UTC+8) `[FACT]`
- **硬件质检台 (Hardware Gate)**：
  - 运行命令：`bash tools/run_qa.sh --recipe configs/profiles/recipe_invar_0.6b_v2.json --iterations 50`
  - 稳态平均前向延迟：**`7.85 ms`** (P50: 7.82 ms, P99: 10.81 ms) `[FACT]`
  - 四大生死红线断言：
    - `mean_latency <= 45.0 ms`: **7.85 ms** (通过)
    - `|choice_sum - 1.0| <= 0.01`: **0.9999** (通过)
    - `1.0 <= impact_score <= 5.0`: **1.91** (通过)
    - `1.0 <= sensitivity_score <= 5.0`: **1.99** (通过)
- **交付产物**：
  - 目录：`/home/zhz/Base-Jev/artifacts/packages/invar-intent-0.6b-v2/`
  - 文件：`system_one_unified.onnx` (1251.12 MB, SHA-256: `1ff03dc66fd8...`), `tokenizer.json`, `MODEL_PASSPORT.json`。`[FACT]`

## 5. Architectural Discoveries & Fixes
1. **ONNX Runtime GPU 约束对齐**：`onnxruntime-gpu` 必须使用 `>=1.20.0,<1.21.0`，严格匹配 CUDA 12.x 与 cuDNN 9.x，杜绝拉入 CUDA 13 专用的 1.30.0。
2. **动态库自动化穿透**：通过 `tools/run_qa.sh` 与 `tools/env_cuda.sh` 自动收集虚拟环境中的 12 个 NVIDIA 动态库路径并注入 `LD_LIBRARY_PATH` 与 `LD_PRELOAD`，彻底杜绝终端环境切换导致的库丢失。
3. **图优化级别对齐**：`crates/core/src/engine.rs` 修正为官方标准支持的 `GraphOptimizationLevel::Level2`（扩展图优化），消除 `graph_optimization_level is not valid` 参数错误。

## 6. Roadmap
- [x] **Phase 1 (已竣工)**：彻底打通 Linux 原生 RTX 3060 CUDA TensorCore 硬件加速，稳态延迟达 7.85 ms，突破生死红线质检并官方签发《模型出厂护照》。
- [ ] **Phase 2 (下一步)**：将出厂标准三件套同步部署至下游工程 `/home/zhz/Invar/models/`。
- [ ] **Phase 3 (最终验收)**：在 Invar 中执行业务端点前向推演测试，完成模型落地闭环。
