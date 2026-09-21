import os
import sys
import argparse
from pathlib import Path

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "src-tauri").is_dir() and (p / "data").is_dir():
            return p
    return Path.cwd().resolve()

PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
TMP_ROOT = PROJECT_ROOT / "tmp"

def download_dataset(repo_id: str, custom_dest: str | None = None) -> Path:
    clean_name = repo_id.split("/")[-1]
    target_dir = Path(custom_dest).resolve() if custom_dest else (DATA_ROOT / clean_name).resolve()
    temp_cache_dir = (TMP_ROOT / "dataset_download_cache").resolve()

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 通用工业级数据集拉取引擎 (Hugging Face / 镜像极速加速)")
    print(f"📦 目标数据集 : {repo_id}")
    print(f"📁 目标落盘   : {target_dir}")
    print(f"📁 缓存隔离   : {temp_cache_dir}")
    print("=" * 70)

    download_success = False

    # Level 1: 国内 HF 镜像站 (https://hf-mirror.com) 极速通道
    print("\n[Level 1] 尝试从国内 Hugging Face 镜像站 (https://hf-mirror.com) 极速拉取...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_CACHE"] = str(temp_cache_dir)
    try:
        from huggingface_hub import snapshot_download as hf_download
        downloaded = hf_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            cache_dir=str(temp_cache_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ [HF 镜像站成功] 数据集资产已完整落盘: {downloaded}")
        download_success = True
    except Exception as exc:
        print(f"⚠️ [HF 镜像站跳过/未命中] 原因: {exc}")

    # Level 2: HF 官方源保底
    if not download_success:
        print("\n[Level 2] 保底方案: 直连 Hugging Face 官方源拉取...")
        os.environ.pop("HF_ENDPOINT", None)
        try:
            from huggingface_hub import snapshot_download as hf_download
            downloaded = hf_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target_dir),
                cache_dir=str(temp_cache_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ [HF 官方源成功] 数据集资产已完整落盘: {downloaded}")
            download_success = True
        except Exception as exc:
            print(f"❌ [下载失败] 两级渠道均未能成功拉取: {exc}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"🎉 目标数据集资产已就绪: {target_dir}")

    total_bytes = 0
    for f in sorted(target_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_bytes += f.stat().st_size
            print(f"  ├─ {f.name:35s} : {size_mb:8.2f} MB")
    print(f"  └─ 实际总物理开销: {total_bytes / (1024 * 1024):.2f} MB")
    print("=" * 70 + "\n")
    return target_dir

def main():
    parser = argparse.ArgumentParser(description="通用工业级数据集拉取同步引擎")
    parser.add_argument(
        "--dataset", 
        default="wtzhang-nlp/fuzzy_bench", 
        help="待拉取的数据集 Repo ID (默认: wtzhang-nlp/fuzzy_bench)"
    )
    parser.add_argument(
        "--dest", 
        default=None, 
        help="自定义落盘路径 (默认: data/<dataset_name>)"
    )
    args = parser.parse_args()
    download_dataset(args.dataset, args.dest)

if __name__ == "__main__":
    main()
