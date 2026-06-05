"""
显著性检测方法可视化对比脚本（筛选前30%候选框，按10%分段着色）
对每张 testA 图像，使用不同显著性方法对候选框评分，筛选出得分前30%的框，
并进一步将筛选出的框按得分分为三段（前10%、10%-20%、20%-30%），用不同颜色绘制。
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))

# 导入队友的候选框生成模块
from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox

# 导入自己的显著性检测器
from saliency.detector import (
   # SpectralResidualDetector, HCDetector, 
    FTDetector
)
from saliency.saliency_utils import extract_bbox_from_framing

# 尝试导入 U2Net
'''try:
    from saliency.detector import U2NetRembgDetector
    HAS_U2NET = True
except:
    HAS_U2NET = False
    print("U2NetRembgDetector 不可用，将跳过")
'''
# ========== 计算 IoU（仅用于标注，不参与筛选）==========
def compute_iou(box1: BBox, box2: BBox) -> float:
    inter_x1 = max(box1.x1, box2.x1)
    inter_y1 = max(box1.y1, box2.y1)
    inter_x2 = min(box1.x2, box2.x2)
    inter_y2 = min(box1.y2, box2.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = box1.area
    area2 = box2.area
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

# ========== 评分策略（使用 center_bias）==========
def score_by_center_bias(sal_map, bbox, center_bias=0.3):
    roi = sal_map[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    if roi.size == 0:
        return 0.0
    mean_score = roi.mean()
    total = sal_map.sum()
    if total == 0:
        return mean_score
    h, w = sal_map.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx_sal = (x_coords * sal_map).sum() / total
    cy_sal = (y_coords * sal_map).sum() / total
    cx_box = (bbox.x1 + bbox.x2) / 2
    cy_box = (bbox.y1 + bbox.y2) / 2
    diag = np.hypot(w, h)
    dist = np.hypot(cx_sal - cx_box, cy_sal - cy_box) / diag
    center_score = 1.0 - dist
    return mean_score * (1 - center_bias) + center_score * center_bias

def filter_top_bboxes_by_percentile(sal_map, candidates, top_percent=0.3, num_segments=3):
    """
    筛选出得分前 top_percent 的候选框，并返回分段索引。
    返回: (top_bboxes, top_scores, segment_indices)
    segment_indices: 列表，每个框对应的分段号（0,1,2）
    """
    scored = [(bbox, score_by_center_bias(sal_map, bbox)) for bbox in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(scored) * top_percent))
    top_bboxes = [bbox for bbox, _ in scored[:k]]
    top_scores = [score for _, score in scored[:k]]
    # 分段：每段长度 k/3，最后一段可能多几个
    seg_len = k // num_segments
    segment_indices = []
    for i in range(k):
        if i < seg_len:
            seg = 0      # 前10%（第一段）
        elif i < 2 * seg_len:
            seg = 1      # 中间10%（第二段）
        else:
            seg = 2      # 最后10%（第三段）
        segment_indices.append(seg)
    return top_bboxes, top_scores, segment_indices

# ========== 加载 testA 数据 ==========
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

# ========== 辅助函数：将显著图叠加到图像 ==========
def overlay_saliency_on_image(img_rgb, sal_map, alpha=0.4):
    sal_uint8 = (sal_map * 255).clip(0, 255).astype(np.uint8)
    sal_3ch = cv2.cvtColor(sal_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, sal_3ch, alpha, 0)
    return overlay

# ========== 主函数 ==========
def main():
    testA_dir = "./data/testA"          # 修改为你的实际路径
    output_dir = "./data/output/saliency_top30%_results"
    os.makedirs(output_dir, exist_ok=True)

    pairs = load_testA_pairs(testA_dir)
    print(f"成功加载 {len(pairs)} 张图像")

    # 准备所有检测器
    detectors = [
      #  SpectralResidualDetector(),
      #  HCDetector(),
        FTDetector(),
    ]
   # if HAS_U2NET:
   #     detectors.append(U2NetRembgDetector())
    # 筛选比例（前30%）
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
                # 筛选前30%框，并分段
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

                # 左图：原图+GT+候选框（分段着色）
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"{img_name} - {det.name} (Top {len(top_bboxes)}/{total_candidates})")
                axes[0].axis('off')
                # 老师真实框（绿色，加粗）
                gt_rect = plt.Rectangle(
                    (gt_bbox.x1, gt_bbox.y1), gt_bbox.width, gt_bbox.height,
                    fill=False, edgecolor='green', linewidth=3, label='GT (Teacher)'
                )
                axes[0].add_patch(gt_rect)

                # 颜色定义：分段0（前10%）红色，分段1（10%-20%）蓝色，分段2（20%-30%）橙色
                segment_colors = ['red', 'dodgerblue', 'orange']
                segment_names = ['Top 10%', 'Middle 10%', 'Bottom 10%']
                # 线宽：高分段2.5，中分段2.0，低分段1.5
                linewidths = [2.5, 2.0, 1.5]

                # 记录已添加的图例，避免重复
                legend_added = set()

                for idx, (bbox, score, seg) in enumerate(zip(top_bboxes, top_scores, seg_ids)):
                    color = segment_colors[seg]
                    lw = linewidths[seg]
                    # 图例：每个分段只显示一次，且前三个框额外显示得分
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

                # 右图：显著图叠加
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