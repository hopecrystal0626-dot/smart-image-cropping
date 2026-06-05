"""
显著性检测方法可视化对比脚本（筛选前30%候选框，按10%分段着色）
使用 saliency_utils 中的评分和筛选函数。
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))

from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox
from saliency.detector import FTDetector
from saliency.saliency_utils import (
    extract_bbox_from_framing,
    compute_iou,
    filter_top_bboxes_by_percentile
)

# 辅助函数：显著图叠加（仅用于可视化）
def overlay_saliency_on_image(img_rgb, sal_map, alpha=0.4):
    sal_uint8 = (sal_map * 255).clip(0, 255).astype(np.uint8)
    sal_3ch = cv2.cvtColor(sal_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, sal_3ch, alpha, 0)
    return overlay

# 加载 testA 数据
def load_testA_pairs(testA_dir):
    pairs = []
    testA_path = Path(testA_dir)
    for img_path in testA_path.glob("*.jpg"):
        if "_framing" in img_path.name:
            continue
        framing_path = testA_path / f"{img_path.stem}_framing.jpg"
        if not framing_path.exists():
            continue
        bbox = extract_bbox_from_framing(str(img_path), str(framing_path))
        if bbox is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_bbox = BBox(bbox[0], bbox[1], bbox[2], bbox[3], scale=1.0)
        pairs.append((img_path.name, img, gt_bbox))
    return pairs

def main():
    testA_dir = "./data/testA"
    output_dir = "./data/output/saliency_top30%_results"
    os.makedirs(output_dir, exist_ok=True)

    pairs = load_testA_pairs(testA_dir)
    print(f"成功加载 {len(pairs)} 张图像")

    detectors = [FTDetector()]   # 当前仅测试 FT
    TOP_PERCENT = 0.3

    for img_name, img, gt_bbox in pairs:
        h, w = img.shape[:2]
        candidates = generate_candidates(w, h)
        if not candidates:
            print(f"警告: {img_name} 无候选框，跳过")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        total_candidates = len(candidates)
        print(f"\n处理 {img_name} (总候选框数: {total_candidates})")

        for det in detectors:
            print(f"  {det.name} ...")
            try:
                sal_map = det.detect(img)
                top_bboxes, top_scores, seg_ids = filter_top_bboxes_by_percentile(
                    sal_map, candidates, TOP_PERCENT, num_segments=3
                )
                if not top_bboxes:
                    print(f"    无有效框")
                    continue

                best_bbox = top_bboxes[0]
                iou = compute_iou(best_bbox, gt_bbox)

                # 绘图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # 左图：原图 + GT + 筛选框
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"{img_name} - {det.name} (Top {len(top_bboxes)}/{total_candidates})")
                axes[0].axis('off')
                gt_rect = plt.Rectangle(
                    (gt_bbox.x1, gt_bbox.y1), gt_bbox.width, gt_bbox.height,
                    fill=False, edgecolor='green', linewidth=3, label='GT (Teacher)'
                )
                axes[0].add_patch(gt_rect)

                segment_colors = ['red', 'dodgerblue', 'orange']
                segment_names = ['Top 10%', 'Middle 10%', 'Bottom 10%']
                linewidths = [2.5, 2.0, 1.5]
                legend_added = set()

                for idx, (bbox, score, seg) in enumerate(zip(top_bboxes, top_scores, seg_ids)):
                    color = segment_colors[seg]
                    lw = linewidths[seg]
                    group_name = segment_names[seg]
                    label = None
                    if group_name not in legend_added:
                        label = group_name
                        legend_added.add(group_name)
                    if idx < 3:
                        label = f"Top{idx+1} (score={score:.3f})"
                    rect = plt.Rectangle(
                        (bbox.x1, bbox.y1), bbox.width, bbox.height,
                        fill=False, edgecolor=color, linewidth=lw,
                        linestyle='-', label=label
                    )
                    axes[0].add_patch(rect)
                axes[0].legend(loc='upper right', fontsize=8)

                # 中图：显著图
                axes[1].imshow(sal_map, cmap='hot')
                axes[1].set_title(f"{det.name} Saliency Map")
                axes[1].axis('off')

                # 右图：叠加图
                overlay = overlay_saliency_on_image(img_rgb, sal_map, alpha=0.4)
                axes[2].imshow(overlay)
                axes[2].set_title(f"Overlay (Best IoU={iou:.3f})")
                axes[2].axis('off')

                out_path = os.path.join(output_dir, f"{img_name}_{det.name}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    已保存 {len(top_bboxes)} 个框 → {out_path}")

            except Exception as e:
                print(f"    错误: {e}")
                continue

    print(f"\n可视化完成，结果保存在 {output_dir}")

if __name__ == "__main__":
    main()