import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src-tauri" / "python" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import onnx
from onnx import external_data_helper

ONNX_PATH = PROJECT_ROOT / "models" / "system_one_unified.onnx"
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-1.7B-Base"


def main():
    print("=" * 70)
    print(" 🔍 ONNX 静态计算图显微镜深度审计")
    print(f" 📦 待测文件: {ONNX_PATH}")
    print("=" * 70)

    if not ONNX_PATH.is_file():
        print(f"[❌] 错误: 文件不存在: {ONNX_PATH}")
        return

    # 1. 查看基模目录下的物理文件
    print("\n[1] 检查本地 models/Qwen3-1.7B-Base 实际文件构成:")
    total_base_size = 0
    for f in BASE_MODEL_DIR.glob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_base_size += f.stat().st_size
            print(f"  ├─ {f.name:35s} : {size_mb:8.2f} MB")
    print(f"  └─ 基模实际总磁盘占用: {total_base_size / (1024*1024*1024):.2f} GB")

    # 2. 解析 ONNX 计算图结构 (不直接加载外部数据)
    print("\n[2] 解析 system_one_unified.onnx Protobuf 计算图:")
    model = onnx.load(str(ONNX_PATH), load_external_data=False)
    graph = model.graph

    print(f"  ├─ 计算图节点总数 (Nodes)      : {len(graph.node)} 个")
    print(f"  ├─ 初始权重张量总数 (Initializers): {len(graph.initializer)} 个")
    print(f"  ├─ 图输入接口数 (Inputs)       : {len(graph.input)} 个")
    print(f"  └─ 图输出接口数 (Outputs)      : {len(graph.output)} 个")

    # 3. 统计初始权重体积与是否存在外部引用
    external_count = 0
    external_locations = set()
    internal_bytes = 0

    for init in graph.initializer:
        if external_data_helper.uses_external_data(init):
            external_count += 1
            for entry in init.external_data:
                if entry.key == "location":
                    external_locations.add(entry.value)
        else:
            if init.raw_data:
                internal_bytes += len(init.raw_data)

    print(f"\n[3] 权重存储分布拓扑:")
    print(f"  ├─ 内存/内嵌张量体积: {internal_bytes / (1024*1024):.2f} MB")
    print(f"  ├─ 外部数据引用数量: {external_count} 个")
    if external_locations:
        print(f"  └─ 外部数据引用的物理文件路径: {list(external_locations)}")
    else:
        print("  └─ 结论: 该 ONNX 文件未声明任何外部数据引用！")

    # 4. 抽样查看节点类型构成
    node_types = {}
    for n in graph.node:
        node_types[n.op_type] = node_types.get(n.op_type, 0) + 1
    print(f"\n[4] 核心算子分布 (Top 10):")
    for op, cnt in sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  ├─ {op:20s}: {cnt} 个")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
