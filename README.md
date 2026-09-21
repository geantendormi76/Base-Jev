# base-jev

基于工业级 Monorepo 对称双引擎架构（对齐 dmtrKovalenko/fff 哲学）的通用工程模板。

## 核心法则与依赖同步
- **Rust 编译**: `cargo check --manifest-path src-tauri/Cargo.toml`
- **Python 依赖同步**: 必须进入 `src-tauri/python` 目录执行 `uv sync`，严禁使用 pip！
- **AI 跨会话上下文交接提取**: 在项目根目录下执行 `npx repomix -c c/1_.json`，即可一键生成高纯净度 `c/repomix-output.xml`

## 目录布局
- `c/`: 认知中枢、解题笔记、知识元数据与 Repomix 交接配置 (`c/1_.json`)
- `data/`: 只读数据集与测试资产
- `models/`: 本地大模型巨型权重金库 (GGUF, ONNX, Safetensors)
- `tmp/`: AI 临时手术台与探索脚本 (随时可清空)
- `src/`: 前端与表现层 (Vue3 / TypeScript)
- `src-tauri/crates/`: Rust 高性能计算引擎
- `src-tauri/python/`: uv 管理的 Python 核心计算服务 (以 `uv sync` 锁死)
- `scripts/pipeline/`: 生产级数据流水线与批量处理
- `scripts/audit/`: 契约审计、数据质检与健康雷达
- `tests/`: 全局端到端 E2E 验收测试
