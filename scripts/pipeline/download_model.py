

# uv run --project src-tauri/python python scripts/pipeline/download_model.py --model Qwen/Qwen3.5-2B-Base

import os
import sys
import argparse
from pathlib import Path

# 弹性解析项目根目录：向上寻找包含 src-tauri 的目录
def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "src-tauri").is_dir() and (p / "models").is_dir():
            return p
    # 默认兜底
    return Path.cwd().resolve()

PROJECT_ROOT = find_project_root()
MODELS_ROOT = PROJECT_ROOT / "models"
TMP_ROOT = PROJECT_ROOT / "tmp"

def download_model(model_id: str, custom_dest: str | None = None) -> Path:
    # 清理可能干扰代理流向的系统环境变量
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        os.environ.pop(var, None)

    clean_name = model_id.split("/")[-1]
    target_dir = Path(custom_dest).resolve() if custom_dest else (MODELS_ROOT / clean_name).resolve()
    temp_cache_dir = (TMP_ROOT / "model_download_cache").resolve()

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 通用三级加速模型拉取引擎")
    print(f"📦 目标模型: {model_id}")
    print(f"📁 目标落盘: {target_dir}")
    print(f"📁 缓存隔离: {temp_cache_dir}")
    print("=" * 70)

    download_success = False

    # Level 1: 阿里魔搭社区 (ModelScope) - 国内首选直连极速源
    print("\n[Level 1] 尝试从阿里魔搭社区 (ModelScope) 极速拉取...")
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
        downloaded = ms_download(
            model_id=model_id,
            local_dir=str(target_dir),
            cache_dir=str(temp_cache_dir),
            revision="master",
        )
        print(f"✓ [ModelScope 成功] 资产已完整落盘: {downloaded}")
        download_success = True
    except Exception as exc:
        print(f"⚠️ [ModelScope 跳过/未命中] 原因: {exc}")

    # Level 2: 国内 HF 镜像站 (https://hf-mirror.com)
    if not download_success:
        print("\n[Level 2] 自动切入国内 Hugging Face 镜像站 (https://hf-mirror.com)...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_CACHE"] = str(temp_cache_dir)
        try:
            from huggingface_hub import snapshot_download as hf_download
            downloaded = hf_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                cache_dir=str(temp_cache_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ [HF 镜像站成功] 资产已完整落盘: {downloaded}")
            download_success = True
        except Exception as exc:
            print(f"⚠️ [HF 镜像站跳过] 原因: {exc}")

    # Level 3: HF 官方源保底
    if not download_success:
        print("\n[Level 3] 保底方案: 直连 Hugging Face 官方源...")
        os.environ.pop("HF_ENDPOINT", None)
        os.environ["HF_HUB_CACHE"] = str(temp_cache_dir)
        try:
            from huggingface_hub import snapshot_download as hf_download
            downloaded = hf_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                cache_dir=str(temp_cache_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ [HF 官方源成功] 资产已完整落盘: {downloaded}")
            download_success = True
        except Exception as exc:
            print(f"❌ [下载失败] 三级渠道均未能成功拉取: {exc}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"🎉 目标模型权重已就绪: {target_dir}")
    print("=" * 70)
    return target_dir

def main():
    parser = argparse.ArgumentParser(description="工业级模型下载同步引擎")
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen3.5-2B-Base", 
        help="待下载的模型 ID (默认: Qwen/Qwen3.5-2B-Base)"
    )
    parser.add_argument(
        "--dest", 
        default=None, 
        help="自定义落盘路径"
    )
    args = parser.parse_args()
    download_model(args.model, args.dest)

if __name__ == "__main__":
    main()



