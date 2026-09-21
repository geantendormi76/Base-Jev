# AI Engineering Monorepo: Repository Inventory & Mapping Plan
# Generated at: S2 INVENTORY
# Canonical Root: C:/dev/Base-Jev

## 1. Current Repository State
- **Git Root**: `C:/dev/Base-Jev`
- **Current Branch**: `main` (HEAD: `1a9dbd1`)
- **Execution Architecture**: Triad Architecture (distiller -> base-jev -> Invar)
- **Current Workspace Condition**: S1 Bootstrap completed, 45 canonical directories active with `.gitkeep` protection.

## 2. Directory Inventory & Boundary Assessment

| Existing Path | Nature / Responsibility | Current File Count | Target Monorepo Domain |
|---|---|---|---|
| `src-tauri/crates/` | Rust DirectML TensorCore Engine & CLI QA tool | ~6 | `crates/core/` |
| `src-tauri/python/src/` | System-1 Decision Heads, Model, Loss, SequencePacker | 8 | `python/packages/core/` |
| `src-tauri/python/tests/` | Python Unit / Contract Tests | 8 | `python/tests/` |
| `recipes/` | Production Recipes (`recipe_invar_0.6b.json`) | 1 | `configs/profiles/` & `configs/base/` |
| `scripts/audit/` | Forensic & Lineage Audit Scripts (E0~E8) | 18 | `scripts/audit/` |
| `scripts/pipeline/` | Training, Export & Distillation Pipeline Scripts | 7 | `python/scripts/` & `scripts/data/` |
| `models/` | Base weights, Checkpoints, and Exported ONNX | ~6 assets | `models/cache/`, `models/checkpoints/`, `artifacts/exports/` |
| `dist/` | Production release bundle (`invar-intent-0.6b-v1`) | 2 assets | `artifacts/packages/` |
| `c/HANDOFF.md` | Session recovery & engineering state snapshot | 1 | `docs/ai/HANDOFF.md` |
| `tools/bootstrap/` | Monorepo scaffolding scripts | 1 | `tools/bootstrap/` |

## 3. Entry Points (入口清单)
- **Rust Engine**: `src-tauri/crates/src/main.rs` (Binary target `base-jev-qa`)
- **Python Trainer**: `scripts/pipeline/train_factory_model.py`
- **ONNX Exporter**: `scripts/pipeline/export_to_onnx.py`
- **Triage Predictor**: `scripts/pipeline/generate_triage_predictions.py`
- **Forensic Auditor**: `scripts/audit/audit_experiment_lineage_v1.py`

## 4. Path Findings & Critical Risks (路径幽灵与硬编码扫描)
- 发现跨工程绝对路径硬编码：
  - `C:/dev/distiller`: 散见于 `scripts/audit/*.py`
  - `C:/dev/Invar`: 散见于 `generate_triage_predictions.py`、`audit_experiment_lineage_v1.py`、`preprocess_and_score_100.py`
  - `C:/dev/base-jev`: 散见于 `audit_experiment_lineage_v1.py`
- 裁决原则：在 S4/S5 阶段必须将这些硬编码路径抽象为 CLI 参数与 Typed Config，严禁硬编码侵入核心包 `python/packages/core/`。

## 5. Potential Conflicts & Resolution (潜在冲突与裁决预案)
- `src-tauri/` 双层嵌套历史包袱：既有 `src-tauri/crates` 与 `src-tauri/python` 混放，需在 S4 分流至顶级 `crates/core` 与 `python/packages/core`。
- `HANDOFF.md` 事实源位置：`c/HANDOFF.md` 需迁移至官方规范路径 `docs/ai/HANDOFF.md`。
- Python 项目依赖配置：`src-tauri/python/pyproject.toml` 需收拢至 `python/pyproject.toml`。
