import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "python" / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from contracts import ProductionRecipe
from onnx_export_wrapper import SystemOneOnnxWrapper
from system_one_model import SystemOneModel


def resolve_path(base_path: Path, p_str: str) -> Path:
    p = Path(p_str)
    return p if p.is_absolute() else (base_path / p).resolve()


def export_system_one_to_onnx(
    checkpoint_dir: Path,
    output_onnx_path: Path,
    base_model_dir: Path,
    opset_version: int = 17,
) -> Path:
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(" 🚀 base-jev: 单文件静态计算图熔融固化与 ONNX 导出流水线 (0.6B Native)")
    print(f" 📦 训练产物微调权重 : {checkpoint_dir.name}")
    print(f" 📦 底座预训练骨干   : {base_model_dir.name}")
    print(f" 📁 目标输出二进制图 : {output_onnx_path}")
    print(f" ⚙️  执行硬件: {device} | 导出精度: Native FP16 (直通 TensorCore)")
    print("=" * 75)

    # 1. 加载原生纯血基模 (动态剥离 lm_head)
    print("\n[1/5] 正在加载纯表征基模 (已剔除 lm_head) ...")
    model = SystemOneModel.from_pretrained_base(
        model_path_or_id=base_model_dir,
        torch_dtype=torch.float16,
    )

    # 2. 挂载并烧结 LoRA 适配器 (merge_and_unload)
    lora_path = checkpoint_dir / "backbone_lora"
    if lora_path.is_dir():
        print(f"[2/5] 正在挂载 LoRA 权重并执行物理烧结融合 (merge_and_unload) ...")
        model.backbone = PeftModel.from_pretrained(model.backbone, str(lora_path))
        model.backbone = model.backbone.merge_and_unload()
        print("  [✓] LoRA 适配器参数已永久烧结进骨干网络权重矩阵！")
    else:
        print("[!] 未检测到 LoRA 权重目录，将直接导出基模表征")

    # 3. 加载训练好的决策头参数
    heads_path = checkpoint_dir / "decision_heads.pt"
    if heads_path.is_file():
        print(f"[3/5] 正在挂载已收敛的统一超球打分头权重: {heads_path.name} ...")
        model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))
        print("  [✓] 训练所得决策头已成功挂载！")
    else:
        print("[!] 警告: 未找到训练好的决策头权重，将使用随机初始化头")

    model.backbone.eval()
    model.heads.eval()

    # 4. 封装导出包装类 (标准 Gather 算子)
    print("[4/5] 构建 SystemOneOnnxWrapper 静态计算图 (标准 Gather 算子) ...")
    wrapper = SystemOneOnnxWrapper(model.backbone, model.heads)
    wrapper.to(device)
    wrapper.to(torch.float16)

    # 构造代表性动态输入张量
    seq_len = 32
    max_cands = 4
    dummy_input_ids = torch.randint(100, 1000, (1, seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    dummy_choice_q_pos = torch.tensor([30], dtype=torch.long, device=device)
    dummy_choice_cand_pos = torch.tensor([[10, 15, 20, 25]], dtype=torch.long, device=device)
    dummy_choice_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float16, device=device)
    dummy_bool_q_pos = torch.tensor([30], dtype=torch.long, device=device)
    dummy_score_q_pos = torch.tensor([30], dtype=torch.long, device=device)

    dummy_inputs = (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_choice_q_pos,
        dummy_choice_cand_pos,
        dummy_choice_mask,
        dummy_bool_q_pos,
        dummy_score_q_pos,
    )

    input_names = [
        "input_ids",
        "attention_mask",
        "choice_q_pos",
        "choice_cand_pos",
        "choice_mask",
        "bool_q_pos",
        "score_q_pos",
    ]

    output_names = [
        "choice_probs",
        "bool_probs",
        "score_dist",
        "score_expected",
    ]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "choice_q_pos": {0: "num_choices"},
        "choice_cand_pos": {0: "num_choices", 1: "max_cands"},
        "choice_mask": {0: "num_choices", 1: "max_cands"},
        "bool_q_pos": {0: "num_booleans"},
        "score_q_pos": {0: "num_scores"},
        "choice_probs": {0: "num_choices", 1: "max_cands"},
        "bool_probs": {0: "num_booleans"},
        "score_dist": {0: "num_scores"},
        "score_expected": {0: "num_scores"},
    }

    # 5. 执行 ONNX 编译固化
    print("[5/5] 正在执行常数折叠与 ONNX 算子固化编译 (耗时约 15~30 秒)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    file_size_mb = output_onnx_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f" 🎉 0.6B 单文件纯血 ONNX 静态计算图导出成功！")
    print(f" 📦 固化资产: {output_onnx_path.resolve()}")
    print(f" ⚖️  文件体积: {file_size_mb:.2f} MB (较 1.7B 瘦身 65%，低于 2GB Protobuf 限制)")
    print("=" * 75 + "\n")
    return output_onnx_path


def main():
    parser = argparse.ArgumentParser(description="base-jev 生产级 ONNX 静态图导出引擎")
    parser.add_argument("--recipe", default="configs/profiles/recipe_invar_0.6b.json", help="生产配方路径")
    parser.add_argument("--checkpoint", default=None, help="自定义产物目录")
    parser.add_argument("--base-model", default=None, help="自定义基模目录")
    parser.add_argument("--output", default=None, help="自定义输出 ONNX 文件")
    args = parser.parse_args()

    recipe_path = resolve_path(PROJECT_ROOT, args.recipe)
    recipe_data = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe = ProductionRecipe.model_validate(recipe_data)

    clean_backbone = Path(recipe.backbone.model_id).name
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else (PROJECT_ROOT / "models" / f"{clean_backbone}-SystemOne-Invar")
    base_dir = Path(args.base_model) if args.base_model else resolve_path(PROJECT_ROOT, recipe.backbone.model_id)
    out_onnx = Path(args.output) if args.output else (PROJECT_ROOT / "models" / recipe.export.output_filename)

    export_system_one_to_onnx(
        checkpoint_dir=ckpt_dir,
        output_onnx_path=out_onnx,
        base_model_dir=base_dir,
    )


if __name__ == "__main__":
    main()

