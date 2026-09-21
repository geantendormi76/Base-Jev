# Path Migration Log
# Executed at: S4 MIGRATION

## Migrated Batches
1. `src-tauri/crates/*` -> `crates/core/*` (Rust Engine Core)
2. `src-tauri/python/src/*` -> `python/packages/core/src/*` (System-1 Python Core)
3. `src-tauri/python/tests/*` -> `python/tests/*` (Python Test Suite)
4. `src-tauri/python/pyproject.toml` -> `python/pyproject.toml` (uv Project Config)
5. `scripts/pipeline/*` -> `python/scripts/*`, `scripts/data/*`, `scripts/build/*` (Scripts Disaggregation)
6. `recipes/*` -> `configs/profiles/*` (Production Configurations)
7. `c/HANDOFF.md` -> `docs/ai/HANDOFF.md` (Canonical AI Handoff Document)
8. Created Root `Cargo.toml` Workspace pointing to `crates/core`.
