# AI Engineering Monorepo: File Decisions & Action Matrix
# Generated at: S3 MAPPING
# Standard Table Schema: Current Path | Type | Responsibility | Target Path | Action | Reason | References | Git Evidence | Risk

| Current Path | Type | Responsibility | Target Path | Action | Reason | References | Git Evidence | Risk |
|---|---|---|---|---|---|---|---|---|
| `c/HANDOFF.md` | Doc | 跨会话工程状态交接唯一事实源 | `docs/ai/HANDOFF.md` | MOVE | 归入母仓库统一规范文档目录 | 项目全局 | Active | Low |
| `src-tauri/Cargo.toml` | Config | 历史 Rust 子工作区配置 | `Cargo.toml` | MERGE | 提升至顶级 Monorepo Cargo Workspace | Rust 全局 | Active | Medium |
| `src-tauri/crates/Cargo.toml` | Config | Core Crate 依赖与清单配置 | `crates/core/Cargo.toml` | MOVE | 收纳至母仓库顶级 crates/core | Rust Engine | Active | Low |
| `src-tauri/crates/src/lib.rs` | Rust | 核心库入口与模块导出 | `crates/core/src/lib.rs` | MOVE | 匹配标准 crates/core 布局 | base_jev_core | Active | Low |
| `src-tauri/crates/src/main.rs` | Rust | DirectML 硬件质检 CLI 二进制入口 | `crates/core/src/main.rs` | MOVE | 属于 Core Crate 的出厂质检工具 | base-jev-qa | Active | Low |
| `src-tauri/crates/src/contracts.rs` | Rust | 纯 Rust 强类型契约与护照数据结构 | `crates/core/src/contracts.rs` | MOVE | 数据与契约边界收敛 | lib.rs, main.rs | Active | Low |
| `src-tauri/crates/src/engine.rs` | Rust | DirectML ONNX TensorCore 运行时驱动 | `crates/core/src/engine.rs` | MOVE | 核心推理引擎实现 | lib.rs, main.rs | Active | Low |
| `src-tauri/crates/tests/contract_test.rs` | Test | Rust 强类型 Serde 往返契约测试 | `crates/core/tests/contract_test.rs` | MOVE | 归入 core crate 测试套件 | Cargo test | Active | Low |
| `src-tauri/python/pyproject.toml` | Config | Python 依赖与 uv 工作区配置 | `python/pyproject.toml` | MOVE | 收纳至顶级 python 工作区根目录 | uv sync, pytest | Active | Medium |
| `src-tauri/python/src/__init__.py` | Python | Python 契约顶层暴露 | `python/packages/core/src/__init__.py` | MOVE | 建立标准 python/packages/core 包 | Python package | Active | Low |
| `src-tauri/python/src/contracts.py` | Python | Pydantic 强类型数据契约与配方规范 | `python/packages/core/src/contracts.py` | MOVE | 核心数据模型边界 | 模块全局引用 | Active | Low |
| `src-tauri/python/src/decision_heads.py` | Python | 超球注意力头与双正交序数打分头实现 | `python/packages/core/src/decision_heads.py` | MOVE | 核心模型架构实现 | model, trainer | Active | Low |
| `src-tauri/python/src/joint_rlcd_loss.py` | Python | Soft-CE + Brier + Entropy 认识论联合损失 | `python/packages/core/src/joint_rlcd_loss.py` | MOVE | 核心训练损失逻辑 | trainer | Active | Low |
| `src-tauri/python/src/sequence_packer.py` | Python | 因果偏序序列打包器与原子锚点管理 | `python/packages/core/src/sequence_packer.py` | MOVE | 核心推断与训练序列编排 | model, scripts | Active | Low |
| `src-tauri/python/src/system_one_model.py` | Python | System-1 骨干网络与 LoRA/Heads 挂载器 | `python/packages/core/src/system_one_model.py` | MOVE | 核心判别式模型底座 | trainer, export | Active | Low |
| `src-tauri/python/src/onnx_export_wrapper.py` | Python | ONNX Gather 算子动态图导出封装 | `python/packages/core/src/onnx_export_wrapper.py` | MOVE | 导出静态图计算算子 | export script | Active | Low |
| `src-tauri/python/src/trainer.py` | Python | 生产工厂微调器与 DataLoader 封装 | `python/packages/core/src/trainer.py` | MOVE | 核心训练循环组件 | train script | Active | Low |
| `src-tauri/python/tests/test_contracts.py` | Test | 数据契约序列化测试 | `python/tests/test_contracts.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `src-tauri/python/tests/test_decision_heads.py` | Test | 决策头数学特性与梯度测试 | `python/tests/test_decision_heads.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `src-tauri/python/tests/test_endpoint_validation.py` | Test | API 防火墙与代码切片提取测试 | `python/tests/test_endpoint_validation.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `src-tauri/python/tests/test_fuzzybench_preprocessing.py` | Test | 通识任务值域聚类测试 | `python/tests/test_fuzzybench_preprocessing.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `src-tauri/python/tests/test_joint_rlcd_loss.py` | Test | 认识论损失数值稳定性测试 | `python/tests/test_joint_rlcd_loss.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `src-tauri/python/tests/test_onnx_export.py` | Test | ONNX Wrapper 前向对齐测试 | `python/tests/test_onnx_export.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `src-tauri/python/tests/test_sequence_packer.py` | Test | 因果偏序与特殊 Token 测试 | `python/tests/test_sequence_packer.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `src-tauri/python/tests/test_system_one_model.py` | Test | lm_head 剥离与特征维度提取测试 | `python/tests/test_system_one_model.py` | MOVE | 标准 Python 测试集 | pytest | Active | Low |
| `recipes/recipe_invar_0.6b.json` | Config | 0.6B 生产训练与导出超参配方 | `configs/profiles/recipe_invar_0.6b.json` | MOVE | 收纳至统一声明式配置目录 | trainer, exporter | Active | Low |
| `scripts/pipeline/train_factory_model.py` | Script | 工厂微调点火训练 CLI 入口 | `python/scripts/train_factory_model.py` | MOVE | 归入 Python 可执行脚本集 | CLI Entry | Active | Low |
| `scripts/pipeline/export_to_onnx.py` | Script | ONNX 算子固化编译导出 CLI 入口 | `python/scripts/export_to_onnx.py` | MOVE | 归入 Python 可执行脚本集 | CLI Entry | Active | Low |
| `scripts/pipeline/generate_triage_predictions.py` | Script | 比较器 1221 神经预测生产脚本 | `python/scripts/generate_triage_predictions.py` | MOVE | 归入 Python 可执行脚本集 | Invar pipeline | Active | Medium |
| `scripts/pipeline/consolidate_onnx.py` | Script | ONNX 权重单文件熔铸压铸脚本 | `scripts/build/consolidate_onnx.py` | MOVE | 归入仓库构建工具集 | Build step | Active | Low |
| `scripts/pipeline/download_dataset.py` | Script | 通用三级加速数据集下载工具 | `scripts/data/download_dataset.py` | MOVE | 归入数据管线工具集 | Data utility | Active | Low |
| `scripts/pipeline/download_model.py` | Script | 通用三级加速模型底座下载工具 | `scripts/data/download_model.py` | MOVE | 归入数据/资产工具集 | Model utility | Active | Low |
| `scripts/pipeline/preprocess_fuzzybench_jev.py` | Script | FuzzyBench 数据集提纯转换工具 | `scripts/data/preprocess_fuzzybench_jev.py` | MOVE | 归入数据预处理工具集 | Data pipeline | Active | Low |
| `scripts/audit/*.py` (18 files) | Script | 取证审计、双师比对与防火墙手术脚本 | `scripts/audit/*.py` | KEEP | 原地保留并规范依赖导入 | Forensic Suite | Active | Low |
| `tools/bootstrap/ai_monorepo_scaffold.py` | Tool | Monorepo 骨架与 .gitkeep 生成器 | `tools/bootstrap/ai_monorepo_scaffold.py` | KEEP | 标准开发者工具 | S1 Bootstrap | Active | Low |
| `README.md` | Doc | 仓库总述与运行指引 | `README.md` | KEEP | 顶层根目录保留，待 S8 刷新 | Root Documentation | Active | Low |
