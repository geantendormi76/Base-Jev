# base-jev (System-1 Cognitive Reflex Model Factory)

> 工业级纯判别式 System-1 决策模型工厂与 DirectML TensorCore 端侧推理引擎。
> 属于 Invar 双引擎系统中的认知快反射前端（~23ms 稳态延迟）。

---

### 1. 工程边界导览 (Directory Map)
- **apps 在哪里？** -> `apps/desktop/`（未来桌面客户端拓展边界）
- **Rust 在哪里？** -> `crates/core/`（包含 DirectML 运行时与 `base-jev-qa` 硬件质检工具）
- **Python 在哪里？** -> `python/packages/core/`（核心决策头、因果偏序打包器、联合损失函数）
- **configs 在哪里？** -> `configs/profiles/`（生产配方如 `recipe_invar_0.6b.json`）
- **models 在哪里？** -> `models/`（基模底座、检查点适配器、ONNX 静态计算图）
- **data 在哪里？** -> `data/`（蒸馏与微调数据集）
- **runs 在哪里？** -> `runs/`（瞬时训练运行日志）
- **artifacts 在哪里？** -> `artifacts/exports/` 与 `dist/`（出厂护照与发布级资产包）
- **tests 在哪里？** -> `python/tests/` 与 `crates/core/tests/`
- **docs 在哪里？** -> `docs/ai/`、`docs/architecture/`、`docs/research/`
- **AI 上下文在哪里？** -> `docs/ai/HANDOFF.md`（唯一事实源）、`docs/ai/REPOSITORY_CLOSURE.md`

### 2. 如何运行与开发 (Quickstart)
- **Python 环境就绪**：
  ```powershell
  cd python
  uv sync
  ```
- **微调点火训练**：
  ```powershell
  uv run python python/scripts/train_factory_model.py --recipe configs/profiles/recipe_invar_0.6b.json
  ```
- **导出单文件静态图 ONNX**：
  ```powershell
  uv run python python/scripts/export_to_onnx.py --recipe configs/profiles/recipe_invar_0.6b.json
  ```
- **Rust 硬件出厂质检**：
  ```powershell
  cargo run --bin base-jev-qa
  ```

### 3. 如何执行全量测试 (Verification)
- **Rust 契约测试**：
  ```powershell
  cargo test --workspace
  ```
- **Python 单元测试**：
  ```powershell
  cd python
  uv run pytest
  ```
