# -*- coding: utf-8 -*-
"""
print_e3_overlap_details.py
只读解析 E3_train_gold_intersection.json 并反查 34 个重叠样本的具体路径与成因
"""
import json
from pathlib import Path

E3_PATH = Path("C:/dev/base-jev/tmp/EXPERIMENT_LINEAGE_AUDIT_V1/E3_train_gold_intersection.json")
GOLD_PATH = Path("C:/dev/Invar/data/targets/ikuai8.com/gold_set_annotated_v1.jsonl")
TRAIN_PATH = Path("C:/dev/base-jev/data/distilled_invar_train.jsonl")

def main():
    print("=" * 85)
    print(" 🔬 正在深度解剖 E3 黄金集 34 处 Surface 碰撞的真实物理样本...")
    print("=" * 85)

    if not E3_PATH.is_file():
        print(f"[❌] 找不到物证文件: {E3_PATH}")
        return

    e3_data = json.loads(E3_PATH.read_text(encoding="utf-8"))
    intersections = e3_data.get("intersections", {})
    
    print("\n[1] E3 物证原始记录摘要:")
    print(f"  • Level 1 Surface 重叠总数 : {intersections.get('level1_surface_id_overlap_count')}")
    print(f"  • Level 2 Route 完整重叠数  : {intersections.get('level2_method_host_path_overlap_count')}")
    print(f"  • Level 3 切片哈希重叠数    : {intersections.get('level3_code_slice_hash_overlap_count')}")
    print(f"  • 训练集 Replay 标签数     : {intersections.get('train_gold_replay_tag_count')}")

    # 加载 Gold Set 和 Train Set 做精准反查
    gold_map = {}
    if GOLD_PATH.is_file():
        with GOLD_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    gold_map[item.get("surface_id", "")] = item

    # 重新在训练集中比对碰撞
    collided_samples = []
    if TRAIN_PATH.is_file():
        with TRAIN_PATH.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    # 按照取证器的算法计算 surface_id
                    from urllib.parse import urlparse
                    import hashlib
                    p = str(item.get("path", "")).strip()
                    m = str(item.get("method", "GET")).upper().strip()
                    parsed = urlparse(p)
                    h = parsed.hostname.lower() if (parsed.scheme and parsed.hostname) else "<unknown-host>"
                    norm_p = parsed.path or "/"
                    if parsed.scheme and parsed.hostname and parsed.query:
                        norm_p = f"{norm_p}?{parsed.query}"
                    sid = hashlib.sha256("\x1f".join([m, h, norm_p]).encode("utf-8")).hexdigest()[:24]

                    if sid in gold_map:
                        collided_samples.append({
                            "train_line": idx,
                            "train_sample_id": item.get("sample_id"),
                            "train_provenance": item.get("provenance"),
                            "train_category": item.get("category"),
                            "train_route": f"{m} {p}",
                            "surface_id": sid,
                            "gold_card_id": gold_map[sid].get("card_id"),
                            "gold_route": f"{gold_map[sid].get('method')} {gold_map[sid].get('path')}",
                        })

    print(f"\n[2] 碰撞样本物理反查详表 (共捕获 {len(collided_samples)} 条具体重叠):")
    print(f"{'Gold 卡片':<12} {'训练集来源':<12} {'类别':<15} {'训练集生成的路由':<40}")
    print("-" * 85)
    for c in collided_samples:
        print(f"[{c['gold_card_id']}] {c['train_provenance']:<12} {c['train_category']:<15} {c['train_route'][:38]:<40}")
        print(f"   └─ 对应 Gold 原题: {c['gold_route']} (Surface: {c['surface_id']})")

    print("\n" + "=" * 85)

if __name__ == "__main__":
    main()
