import argparse
import json
from pathlib import Path
import sys
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "python" / "packages" / "core" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contracts import PrecisionTier, ProductionRecipe
from onnx_export_wrapper import SystemOneOnnxWrapper
from system_one_model import SystemOneModel


def resolve_path(base_path: Path, p_str: str) -> Path:
    p = Path(p_str)
    if p.is_absolute():
        return p
    return (base_path / p).resolve()


def export_onnx(
    recipe: ProductionRecipe,
    checkpoint_dir: Path,
    base_model_dir: Path,
    output_onnx_path: Path,
) -> None:
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_fp16 = recipe.export.precision == PrecisionTier.FP16

    print("===========================================================================")
    print(" 🚀 base-jev: ONNX 静态计算图导出流水线 (真·双正交头 v2.0)")
    print(f" 📦 产物导出路径 : {output_onnx_path}")
    print(f" 🎯 目标执行设备 : {device} (精度: {recipe.export.precision.value})")
    print(f" 📂 基模目录     : {base_model_dir}")
    print(f" 📂 检查点目录   : {checkpoint_dir}")
    print("===========================================================================")

    # 1. 加载原生纯血基模 (动态剥离 lm_head)
    print("\n[1/5] 正在加载纯表征 Transformer 骨干...")
    torch_dtype = torch.float16 if (is_fp16 and device.type == "cuda") else torch.float32
    model = SystemOneModel.from_pretrained_base(
        base_model_dir,
        device_map=None,
        torch_dtype=torch_dtype,
    )
    print(f"  [✓] 骨干表征加载完成 (隐层维度: {model.hidden_size})")

    # 2. 挂载并烧结 LoRA 适配器 (merge_and_unload)
    lora_path = checkpoint_dir / "backbone_lora"
    if lora_path.is_dir():
        print(f"\n[2/5] 正在挂载并烧结 LoRA 权重: {lora_path}")
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(model.backbone, str(lora_path))
        model.backbone = peft_model.merge_and_unload()
        print("  [✓] LoRA 适配器权重已成功融入骨干静态图！")
    else:
        print(f"\n[2/5] 未检测到 LoRA 目录 ({lora_path})，跳过融合步骤。")

    # 3. 加载训练好的决策头参数
    heads_path = checkpoint_dir / "decision_heads.pt"
    if heads_path.is_file():
        print(f"\n[3/5] 正在加载多原语决策头参数: {heads_path}")
        heads_state = torch.load(str(heads_path), map_location=device)
        model.heads.load_state_dict(heads_state, strict=False)
        print("  [✓] 决策头权重加载成功 (已包含 impact 与 sensitivity 双头参数)")
    else:
        print(f"\n[3/5] 警告: 未找到 {heads_path}，使用初始化权重导出！")

    # 统一设备与精度
    model.to(device)
    if is_fp16 and device.type == "cuda":
        model = model.half()
    model.eval()

    # 4. 封装导出包装类 (RFC-BASE-JEV-ONNX-DUAL-HEAD-V2)
    print("\n[4/5] 正在构建物理双头 ONNX 静态计算图包装器...")
    wrapper = SystemOneOnnxWrapper(model.backbone, model.heads)
    wrapper.eval()

    # 构造代表性动态输入张量
    seq_len = 32
    max_cands = 4
    mask_dtype = torch.float16 if (is_fp16 and device.type == "cuda") else torch.float32

    dummy_input_ids = torch.randint(100, 1000, (1, seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    dummy_choice_q_pos = torch.tensor([30], dtype=torch.long, device=device)
    dummy_choice_cand_pos = torch.tensor([[10, 15, 20, 25]], dtype=torch.long, device=device)
    dummy_choice_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=mask_dtype, device=device)
    dummy_bool_q_pos = torch.tensor([30], dtype=torch.long, device=device)
    dummy_impact_q_pos = torch.tensor([30], dtype=torch.long, device=device)
    dummy_sensitivity_q_pos = torch.tensor([30], dtype=torch.long, device=device)

    dummy_inputs = (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_choice_q_pos,
        dummy_choice_cand_pos,
        dummy_choice_mask,
        dummy_bool_q_pos,
        dummy_impact_q_pos,
        dummy_sensitivity_q_pos,
    )

    input_names = [
        "input_ids",
        "attention_mask",
        "choice_q_pos",
        "choice_cand_pos",
        "choice_mask",
        "bool_q_pos",
        "impact_q_pos",
        "sensitivity_q_pos",
    ]

    output_names = [
        "choice_probs",
        "bool_probs",
        "impact_dist",
        "impact_expected",
        "sensitivity_dist",
        "sensitivity_expected",
    ]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "choice_q_pos": {0: "num_choices"},
        "choice_cand_pos": {0: "num_choices", 1: "max_cands"},
        "choice_mask": {0: "num_choices", 1: "max_cands"},
        "bool_q_pos": {0: "num_booleans"},
        "impact_q_pos": {0: "num_impact_queries"},
        "sensitivity_q_pos": {0: "num_sensitivity_queries"},
        "choice_probs": {0: "num_choices", 1: "max_cands"},
        "bool_probs": {0: "num_booleans", 1: "dim_one"},
        "impact_dist": {0: "num_impact_queries", 1: "max_levels"},
        "impact_expected": {0: "num_impact_queries", 1: "dim_one"},
        "sensitivity_dist": {0: "num_sensitivity_queries", 1: "max_levels"},
        "sensitivity_expected": {0: "num_sensitivity_queries", 1: "dim_one"},
    }

    # 5. 执行 ONNX 编译固化
    print(f"\n[5/5] 正在执行 torch.onnx.export 编译固化至 {output_onnx_path} ...")
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    file_size_mb = output_onnx_path.stat().st_size / (1024 * 1024)

    print("===========================================================================")
    print(" 🎉 ONNX 静态计算图导出成功！")
    print(f" 📄 文件物理大小 : {file_size_mb:.2f} MB")
    print(f" ⏱️ 编译耗时     : {elapsed:.2f} 秒")
    print(f" 🎯 物理输出张量 : {output_names}")
    print("===========================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="base-jev 生产级 ONNX 静态图导出引擎")
    parser.add_argument(
        "--recipe",
        type=str,
        required=True,
        help="生产订单配方路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="训练产物检查点目录 (可选)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="基模物理目录 (可选)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="ONNX 产物输出路径 (可选)",
    )
    args = parser.parse_args()

    recipe_path = resolve_path(PROJECT_ROOT, args.recipe)
    recipe_data = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe = ProductionRecipe.model_validate(recipe_data)

    clean_backbone = Path(recipe.backbone.model_id).name
    ckpt_dir = (
        Path(args.checkpoint)
        if args.checkpoint
        else (PROJECT_ROOT / "models" / f"{clean_backbone}-SystemOne-{recipe.recipe_id}")
    )
    base_dir = (
        Path(args.base_model)
        if args.base_model
        else resolve_path(PROJECT_ROOT, recipe.backbone.model_id)
    )
    out_onnx = (
        Path(args.output)
        if args.output
        else (PROJECT_ROOT / "models" / recipe.export.output_filename)
    )

    export_onnx(
        recipe=recipe,
        checkpoint_dir=ckpt_dir,
        base_model_dir=base_dir,
        output_onnx_path=out_onnx,
    )


if __name__ == "__main__":
    main()
