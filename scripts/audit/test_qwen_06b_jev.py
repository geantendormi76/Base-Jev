import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = PROJECT_ROOT / "src-tauri" / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import torch
from system_one_model import SystemOneModel

MODEL_06B_DIR = PROJECT_ROOT / "models" / "Qwen3-0.6B-Base"

def test_qwen_06b_compatibility():
    print("=" * 75)
    print(" 🔬 Qwen3-0.6B-Base 原生底座与 JEV 架构装配健康度探查 (纯 CPU 模式)")
    print(f" 📂 权重路径: {MODEL_06B_DIR}")
    print("=" * 75)

    if not MODEL_06B_DIR.is_dir():
        print(f"[❌] 目录不存在: {MODEL_06B_DIR}")
        return

    # 1. 严格使用 CPU 加载，绝不触碰正在蒸馏的 GPU 显存！
    print("\n[1/3] 正在挂载 0.6B 骨干并动态剔除 lm_head ...")
    model = SystemOneModel.from_pretrained_base(
        model_path_or_id=MODEL_06B_DIR,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    hidden_size = model.hidden_size
    print(f"  ├─ 动态萃取隐层特征维度 (Hidden Size) : {hidden_size} 维")
    print(f"  └─ lm_head 剔除状态                   : {'✅ 已彻底剔除 (零死重)' if not hasattr(model.backbone, 'lm_head') else '❌ 异常残留'}")

    # 2. 检查 JEV 三大多态决策头装配参数
    print("\n[2/3] 检查 JEV 决策头挂载拓扑:")
    print(f"  ├─ Choice 超球头输入维度              : {model.heads.choice_head.hidden_size} 维 (投影子空间: {model.heads.choice_head.projection_dim})")
    print(f"  ├─ Boolean 标量头输入维度             : {model.heads.boolean_head.classifier.in_features} 维")
    print(f"  └─ Score 序数头输入维度               : {model.heads.score_head.classifier.in_features} 维 (最大档位: {model.heads.score_head.max_levels})")

    # 3. 模拟纯前向张量演练 (8 个 Token，单步前向)
    print("\n[3/3] 执行纯 CPU 单步前向推演演练...")
    dummy_input = torch.tensor([[101, 200, 300, 400, 500, 600, 700, 800]], dtype=torch.long)
    with torch.no_grad():
        hidden_states = model.extract_hidden_states(dummy_input)

    print(f"  ├─ 骨干特征输出张量形状 (Batch, Seq, Dim): {tuple(hidden_states.shape)}")
    assert hidden_states.shape == (1, 8, hidden_size)

    # 演练超球打分头 (3 个候选分支)
    q_vec = hidden_states[:, -1, :]
    c_vecs = hidden_states[:, :3, :]
    _, choice_probs = model.heads.forward_choice(q_vec, c_vecs)
    prob_sum = float(choice_probs.sum())
    print(f"  ├─ Choice 选项概率和归一化核验           : {prob_sum:.4f} (要求为 1.0)")
    assert abs(prob_sum - 1.0) < 1e-4

    print("\n" + "=" * 75)
    print(" 🎉 0.6B 轻量骨干与 JEV 纯前向决策架构 100% 完美兼容，毫无缝隙！")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    test_qwen_06b_compatibility()
