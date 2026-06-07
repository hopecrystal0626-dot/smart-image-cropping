# experiments/test_single_image.py
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox
from saliency.detector import FTDetector
from saliency.saliency_utils import filter_top_bboxes_by_percentile, get_subject_center
from composition.aesthetic_scorer import AestheticScorer
from composition.human_detector import HumanDetector
from composition.object_detector import ObjectDetector

def compute_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def expand_bbox_to_cover_subject(bbox, subject_box, img_w, img_h, expand_ratio=0.15):
    x1 = min(bbox.x1, subject_box[0]); y1 = min(bbox.y1, subject_box[1])
    x2 = max(bbox.x2, subject_box[2]); y2 = max(bbox.y2, subject_box[3])
    w = x2 - x1; h = y2 - y1
    dx = int(w * expand_ratio); dy = int(h * expand_ratio)
    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
    x2 = min(img_w, x2 + dx); y2 = min(img_h, y2 + dy)
    return BBox(x1, y1, x2, y2, bbox.scale)

def resize_bbox_to_target_area(bbox, target_ratio, img_w, img_h):
    area_ratio = bbox.area / (img_w * img_h)
    if abs(area_ratio - target_ratio) < 0.01:
        return bbox
    scale = np.sqrt(target_ratio / area_ratio)
    new_w = int(bbox.width * scale)
    new_h = int(bbox.height * scale)
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    x1 = int(cx - new_w/2)
    y1 = int(cy - new_h/2)
    x2 = x1 + new_w
    y2 = y1 + new_h
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h
    return BBox(x1, y1, x2, y2, bbox.scale)

def process_single_image(img_path, output_dir="./data/output/single_test"):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    img_name = Path(img_path).stem
    h, w = img.shape[:2]
    total_area = w * h

    # 初始化组件
    saliency_detector = FTDetector()
    aesthetic_scorer = AestheticScorer()
    mtcnn_detector = HumanDetector()
    yolo = ObjectDetector(conf_threshold=0.25)

    # 参数
    TOP_PERCENT_SAL = 0.3
    TOP_PERCENT_AES = 0.05
    W_HUMAN = 1.0
    W_OBJECT = 0.8
    W_CENTER = 0.2
    MIN_COVER_THRESH = 0.3
    EXPAND_RATIO = 0.15
    TARGET_AREA_RATIO = 0.3
    AREA_LOW = 0.2
    AREA_HIGH = 0.5

    # 候选框生成
    candidates = generate_candidates(w, h)
    if not candidates:
        print("未生成候选框")
        return

    # 显著性检测
    sal_map = saliency_detector.detect(img)
    top_bboxes, _, _ = filter_top_bboxes_by_percentile(sal_map, candidates, top_percent=TOP_PERCENT_SAL)
    if not top_bboxes:
        print("显著性筛选后无候选框")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 人体和物体检测
    human_boxes, object_boxes = yolo.detect_all(img_rgb, verbose=True)
    if not human_boxes:
        human_boxes = mtcnn_detector.detect_human_bboxes(img_rgb)
        if human_boxes:
            print(f"    MTCNN 检测到人脸: {len(human_boxes)}")

    # 主体框
    expand_box = None
    if human_boxes:
        expand_box = max(human_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    elif object_boxes:
        expand_box = max(object_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

    # 目标中心
    if expand_box:
        target_cx = (expand_box[0] + expand_box[2]) / 2
        target_cy = (expand_box[1] + expand_box[3]) / 2
    else:
        target_cx, target_cy = get_subject_center(sal_map)

    # 评分
    scored = []
    for bbox in top_bboxes:
        aes = aesthetic_scorer.get_score_for_bbox(img, bbox)
        human_cover = max([compute_iou_xyxy((bbox.x1, bbox.y1, bbox.x2, bbox.y2), hb) for hb in human_boxes]) if human_boxes else 0.0
        obj_cover = max([compute_iou_xyxy((bbox.x1, bbox.y1, bbox.x2, bbox.y2), ob) for ob in object_boxes]) if object_boxes else 0.0
        cover = W_HUMAN * human_cover + W_OBJECT * obj_cover
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        dist = np.hypot(cx - target_cx, cy - target_cy)
        max_dist = np.hypot(w, h) / 2
        center_score = 1.0 - min(1.0, dist / max_dist)
        final = aes + cover + W_CENTER * center_score
        scored.append((bbox, final, aes, cover, center_score, human_cover, obj_cover))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 保底
    if expand_box:
        best_cover = max(scored, key=lambda x: x[3])
        if scored[0][3] < MIN_COVER_THRESH and best_cover[3] > MIN_COVER_THRESH:
            print(f"  保底替换: {scored[0][3]:.2f} -> {best_cover[3]:.2f}")
            scored.remove(best_cover)
            scored.insert(0, best_cover)

    k = max(1, int(len(scored) * TOP_PERCENT_AES))
    final_bboxes_raw = [x[0] for x in scored[:k]]
    final_scores_raw = [x[1] for x in scored[:k]]

    # 扩展并面积约束
    final_bboxes = []
    final_scores = []
    for idx, (bbox, score) in enumerate(zip(final_bboxes_raw, final_scores_raw)):
        if idx == 0 and expand_box:
            if compute_iou_xyxy((bbox.x1, bbox.y1, bbox.x2, bbox.y2), expand_box) > 0.1:
                bbox = expand_bbox_to_cover_subject(bbox, expand_box, w, h, EXPAND_RATIO)
        if idx == 0:
            area_ratio = bbox.area / total_area
            if area_ratio < AREA_LOW or area_ratio > AREA_HIGH:
                print(f"    面积约束: 当前比例 {area_ratio:.2f}，调整为 {TARGET_AREA_RATIO:.2f}")
                bbox = resize_bbox_to_target_area(bbox, TARGET_AREA_RATIO, w, h)
        final_bboxes.append(bbox)
        final_scores.append(score)

    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"{img_name} (top{int(TOP_PERCENT_AES*100)}%)")
    axes[0].axis('off')
    # 人体框（蓝）
    for hb in human_boxes:
        axes[0].add_patch(plt.Rectangle((hb[0], hb[1]), hb[2]-hb[0], hb[3]-hb[1],
                                        fill=False, edgecolor='blue', lw=1.5, label='Human'))
    # 物体框（青）
    for ob in object_boxes:
        axes[0].add_patch(plt.Rectangle((ob[0], ob[1]), ob[2]-ob[0], ob[3]-ob[1],
                                        fill=False, edgecolor='cyan', lw=1.5, label='Object'))
    # 最终框
    for idx, (bbox, s) in enumerate(zip(final_bboxes, final_scores)):
        color = 'yellow' if idx == 0 else 'red'
        lw = 3 if idx == 0 else 1.5
        label = f'Best ({s:.2f})' if idx == 0 else (f'Top{idx+1}' if idx < 3 else '')
        axes[0].add_patch(plt.Rectangle((bbox.x1, bbox.y1), bbox.width, bbox.height,
                                        fill=False, edgecolor=color, lw=lw, label=label))
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].imshow(sal_map, cmap='hot')
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')

    out_path = os.path.join(output_dir, f"{img_name}_single.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"结果保存至: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单张图像智能取景测试")
    parser.add_argument("image_path", type=str, help="图像路径")
    parser.add_argument("--output_dir", type=str, default="./data/output/single_test", help="输出目录")
    args = parser.parse_args()
    process_single_image(args.image_path, args.output_dir)