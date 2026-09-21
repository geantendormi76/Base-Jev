from pathlib import Path

TARGET_DIRS = [
    "apps/desktop/src",
    "apps/desktop/src-tauri/src",
    "apps/desktop/src-tauri/capabilities",
    "apps/desktop/src-tauri/icons",
    "crates/core",
    "python/packages/core",
    "python/scripts",
    "python/tests",
    "configs/base",
    "configs/profiles",
    "configs/local",
    "configs/examples",
    "configs/schemas",
    "configs/registry",
    "data/inbox",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/fixtures",
    "data/manifests",
    "models/registry",
    "models/checkpoints",
    "models/adapters",
    "models/tokenizers",
    "models/cache",
    "runs",
    "artifacts/reports",
    "artifacts/metrics",
    "artifacts/exports",
    "artifacts/packages",
    "tests/integration",
    "tests/e2e",
    "tests/contract",
    "tests/fixtures",
    "scripts/bootstrap",
    "scripts/build",
    "scripts/data",
    "scripts/audit",
    "scripts/release",
    "docs/architecture",
    "docs/research",
    "docs/decisions",
    "docs/ai",
    "tools/bootstrap",
    "tools/repomix",
    "tmp/repomix",
    ".github/workflows",
]

def main():
    root = Path(__file__).resolve().parents[2]
    print(f"[*] AI Engineering Monorepo 正在构建标准骨架: {root}")
    created_dirs = 0
    created_gitkeeps = 0
    for rel in TARGET_DIRS:
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        created_dirs += 1
        # 如果目录为空（无文件也无子目录），必须创建 .gitkeep 确保 Git 边界锁定
        if not any(d.iterdir()):
            gk = d / ".gitkeep"
            gk.touch(exist_ok=True)
            created_gitkeeps += 1
    print(f"[✓] 骨架装配完毕: 确认 {created_dirs} 个标准目录, 补全 {created_gitkeeps} 个必要 .gitkeep")

if __name__ == "__main__":
    main()
