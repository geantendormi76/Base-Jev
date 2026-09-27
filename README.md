An AI engineering monorepo for research, inference, agents, training, data processing,
desktop applications and systems software.

## Repository map

| Path | Responsibility | Lifecycle |
|---|---|---|
| `apps/` | shippable applications / entry points | product |
| `crates/` | reusable Rust engines and domain modules | source |
| `python/` | uv-managed Python workspace | source |
| `configs/` | configuration injection and registries | declarative |
| `data/` | datasets and transformed inputs | data |
| `models/` | model metadata, weights, adapters and caches | large / external |
| `artifacts/` | intentional outputs and reports | generated |
| `docs/` | architecture, research and AI handoff | durable knowledge |
| `tools/` | developer tooling | tooling |
| `tmp/` | disposable workspace | ephemeral |

## Core commands

Rust workspace:

```text
cargo check --workspace
cargo test --workspace
```

Frontend / Tauri:

```text
pnpm install
pnpm desktop:dev
pnpm desktop:build
```

Python workspace:

```text
cd python
uv sync
uv run pytest
```

AI context packaging:

```text
npx repomix -c tools/repomix/1_.json
```

The resulting context bundle is disposable and belongs under `tmp/repomix/`.

## Configuration contract

Configuration is layered as:

`base -> profile -> local -> environment -> CLI`

Backend/model/provider selection belongs in `configs/registry/` and should be connected
to stable adapters rather than implementation-specific branching in orchestration code.