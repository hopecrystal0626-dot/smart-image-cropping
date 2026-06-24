"""
诊断脚本: 打印 instance_masks 里每个实例的面积占比, 并尝试反查对应的语义标签
(通过和 segments 的 mask 做 IoU 匹配), 用于排查 object_clip_penalty
普遍顶格(0.50)的原因 —— 怀疑是面积占比 30%~45% 的"风景类"实例
(草地/水面/建筑群等)被错误地当成"中小型不可截断物体"。

用法:
    python diagnose_instance_masks.py --img A20
"""

import os
import sys
import argparse

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from filter_and_visualize import (
    process_one,
    load_image,
    get_panoptic_result,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="A01")
    parser.add_argument("--no_depth", action="store_true")
    args = parser.parse_args()

    img_id = args.img

    img_rgb = load_image(img_id)
    h, w = img_rgb.shape[:2]
    img_area = h * w

    _, segments, seg_map = get_panoptic_result(img_rgb)

    print(f"===== {img_id} ({w}x{h}) =====")
    print("\nsegments (mask2former):")
    for seg in segments:
        ratio = seg.get("area", 0) / img_area
        print(f"  label={seg['label']:20s} score={seg.get('score', -1):.2f} "
              f"area_ratio={ratio:.3f}")

    # 复用 process_one 拿到 instance_masks
    _, final_records, _, instance_masks, landscape_masks = process_one(img_id, args)

    print(f"\ninstance_masks 共 {len(instance_masks)} 个:")
    for i, m in enumerate(instance_masks):
        ratio = m.sum() / img_area

        # 反查: 和哪个 segment 的 mask 重合度最高
        best_label = None
        best_iou = 0.0
        for seg in segments:
            seg_id = seg.get("id", None)
            if seg_id is not None:
                seg_mask = (seg_map == seg_id).astype(np.uint8)
            else:
                x1, y1, x2, y2 = seg["bbox"]
                seg_mask = np.zeros((h, w), dtype=np.uint8)
                seg_mask[y1:y2, x1:x2] = 1

            inter = np.logical_and(m, seg_mask).sum()
            union = np.logical_or(m, seg_mask).sum()
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_label = seg["label"]

        flag = ""
        if 0.30 <= ratio < 0.45:
            flag = "  <== 面积占比在30%~45%区间, 是object_clip_penalty的'中小物体'判定边缘"

        print(f"  实例{i:2d}: 面积占比={ratio:.3f}  最可能对应label={best_label}"
              f"(IoU={best_iou:.2f}){flag}")


if __name__ == "__main__":
    main()