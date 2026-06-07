"""
完整逻辑：
同学A（滑动窗口）→ 同学B（显著性筛选）→ YOLO完整性检测 → 美学评分 → 选出完整且得分最高的前10个框
"""

import os
import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from composition.composition_score import compute_composition_score_single
from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox
from experiments.test_saliency import (
    FTDetector,
    filter_top_bboxes_by_percentile,
    extract_bbox_from_framing,
    compute_iou
)


# ============================================================
# 配置
# ============================================================
YOLO_MODEL = str(project_root / "yolov8n.pt")
MIN_CONF = 0.3                     # YOLO 检测置信度
COMPLETENESS_THRESHOLD = 0.85      # 物体完整度阈值（0.85表示85%以上被框住才算完整）
TOP_PERCENT = 0.3                  # 显著性筛选保留比例
# ============================================================


class CompletenessChecker:
    """物体完整性检测器（使用YOLO）"""
    
    def __init__(self, model_path=YOLO_MODEL):
        self.model = YOLO(model_path)
        print("✅ YOLO 物体检测模型加载成功")
    
    def detect_all_objects(self, img):
        """检测原图中所有物体"""
        results = self.model(img)
        objects = []
        
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.conf[0] < MIN_CONF:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                cls_name = self.model.names[cls]
                objects.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': cls_name,
                    'conf': float(box.conf[0]),
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        # 按面积降序排序（主要物体优先）
        objects.sort(key=lambda x: x['area'], reverse=True)
        return objects
    
    def check_candidate_completeness(self, candidate_bbox, all_objects):
        """
        检查一个候选框是否完整包含了检测到的物体
        返回: (是否完整, 完整度得分, 包含的主要物体)
        """
        cx1, cy1, cx2, cy2 = candidate_bbox.x1, candidate_bbox.y1, candidate_bbox.x2, candidate_bbox.y2
        c_area = (cx2 - cx1) * (cy2 - cy1)
        
        best_match = None
        best_completeness = 0
        
        for obj in all_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            o_area = obj['area']
            
            # 计算物体被候选框覆盖的比例
            ix1 = max(cx1, ox1)
            iy1 = max(cy1, oy1)
            ix2 = min(cx2, ox2)
            iy2 = min(cy2, oy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                completeness = inter_area / o_area  # 物体被覆盖的比例
                
                if completeness > best_completeness:
                    best_completeness = completeness
                    best_match = obj
        
        # 判断是否完整
        is_complete = best_completeness >= COMPLETENESS_THRESHOLD
        
        return is_complete, best_completeness, best_match


def process_single_image(img_path, framing_path, checker, top_percent=0.3):
    """处理单张图片"""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None, None
    
    h, w = img.shape[:2]
    
    # 获取老师框（仅用于对比评估，不参与决策）
    gt_bbox = None
    if os.path.exists(framing_path):
        bbox = extract_bbox_from_framing(img_path, framing_path)
        if bbox:
            gt_bbox = BBox(bbox[0], bbox[1], bbox[2], bbox[3], scale=1.0)
    
    # ========== 步骤1：检测原图中所有物体（用于完整性判断）==========
    all_objects = checker.detect_all_objects(img)
    if len(all_objects) > 0:
        print(f"   检测到 {len(all_objects)} 个物体: ", end="")
        for obj in all_objects[:5]:
            print(f"{obj['class']} ", end="")
        print()
    else:
        print(f"   未检测到物体，将跳过完整性检查")
    
    # ========== 步骤2：同学A - 生成候选框 ==========
    all_candidates = generate_candidates(w, h)
    print(f"   步骤1-候选框总数: {len(all_candidates)}")
    
    # ========== 步骤3：尺寸筛选 ==========
    MIN_AREA_RATIO = 0.15
    filtered_by_size = []
    for bbox in all_candidates:
        area_ratio = (bbox.width * bbox.height) / (w * h)
        if area_ratio >= MIN_AREA_RATIO:
            filtered_by_size.append(bbox)
    print(f"   步骤2-尺寸筛选后: {len(filtered_by_size)} 个")
    
    # ========== 步骤4：同学B - 显著性筛选 ==========
    detector = FTDetector()
    sal_map = detector.detect(img)
    result = filter_top_bboxes_by_percentile(sal_map, filtered_by_size, top_percent)
    top_bboxes = result[0]
    print(f"   步骤3-显著性筛选后: {len(top_bboxes)} 个")
    
    if len(top_bboxes) == 0:
        return None, None, None, None, None
    
    # ========== 步骤5：完整性检测（只保留完整包含物体的候选框）==========
    complete_boxes = []
    completeness_scores = []
    
    for bbox in top_bboxes:
        is_complete, completeness, matched_obj = checker.check_candidate_completeness(bbox, all_objects)
        
        if len(all_objects) == 0:
            # 没有检测到物体，保留所有框
            complete_boxes.append(bbox)
            completeness_scores.append(1.0)
        elif is_complete:
            complete_boxes.append(bbox)
            completeness_scores.append(completeness)
        else:
            # 调试：打印被剔除的框信息
            if matched_obj:
                print(f"   ⚠️ 剔除: 物体={matched_obj['class']}, 完整度={completeness:.2f}")
    
    print(f"   步骤4-完整性筛选后: {len(complete_boxes)} 个")
    
    # 如果没有完整框，降级使用原始框（但会给出警告）
    if len(complete_boxes) == 0:
        complete_boxes = top_bboxes
        print(f"   ⚠️ 警告：没有完整框，降级使用原始候选框（可能包含截断物体）")
    
    # ========== 步骤6：你的美学评分 ==========
    scored_boxes = []
    for i, bbox in enumerate(complete_boxes):
        cropped = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if cropped.size == 0:
            continue
        
        score_detail = compute_composition_score_single(cropped, bbox=(bbox.x1, bbox.y1, bbox.width, bbox.height))
        comp_score = score_detail['total_score']
        iou = compute_iou(bbox, gt_bbox) if gt_bbox else 0
        
        # 获取完整度得分
        completeness = completeness_scores[i] if i < len(completeness_scores) else 1.0
        
        scored_boxes.append({
            'bbox': bbox,
            'score': comp_score,
            'completeness': completeness,
            'iou': iou,
        })
    
    if len(scored_boxes) == 0:
        return None, None, None, None, None
    
    # 按美学得分排序
    scored_boxes.sort(key=lambda x: x['score'], reverse=True)
    top10 = scored_boxes[:10]
    best = top10[0] if top10 else None
    
    return top10, best, gt_bbox, all_objects, complete_boxes


def visualize_result(img_path, top10, best, gt_bbox, objects, output_path):
    """可视化结果"""
    img = cv2.imread(img_path)
    if img is None:
        return
    
    canvas = img.copy()
    
    # 检测到的物体（蓝色细线，只画前3个）
    for obj in objects[:3]:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(canvas, obj['class'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    
    # 老师框（白色粗线）
    if gt_bbox:
        cv2.rectangle(canvas, (gt_bbox.x1, gt_bbox.y1), (gt_bbox.x2, gt_bbox.y2), (255, 255, 255), 3)
        cv2.putText(canvas, "Teacher", (gt_bbox.x1, gt_bbox.y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 前十名（绿色细线，第一名单独处理）
    for i, item in enumerate(top10):
        if i == 0:
            continue
        bbox = item['bbox']
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
        label = f"#{i+1}: {item['score']:.2f}"
        cv2.putText(canvas, label, (bbox.x1, bbox.y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    # 最佳框（粉色粗线）
    if best:
        bbox = best['bbox']
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 0, 255), 3)
        cv2.putText(canvas, f"Best: {best['score']:.3f}", 
                   (bbox.x1, bbox.y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    cv2.imwrite(output_path, canvas)


def main():
    testA_dir = r"D:\VSCODE project\jiqishijue\smart-image-cropping\data\testA"
    output_dir = r"D:\VSCODE project\jiqishijue\smart-image-cropping\data\output\final_complete"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("智能取景系统 - 完整版")
    print("流程: 同学A → 同学B → 完整性检测 → 美学评分")
    print(f"完整度阈值: {COMPLETENESS_THRESHOLD}")
    print("颜色: 粉色=最佳框, 绿色=其他前十名, 白色=老师框, 蓝色=检测物体")
    print("="*60)
    
    # 初始化完整性检测器
    checker = CompletenessChecker()
    
    # 获取所有测试图片
    testA_path = Path(testA_dir)
    image_files = []
    for img_path in testA_path.glob("*.jpg"):
        if "_framing" in img_path.name:
            continue
        framing_path = testA_path / f"{img_path.stem}_framing.jpg"
        if framing_path.exists():
            image_files.append((img_path, framing_path))
    
    print(f"\n找到 {len(image_files)} 张测试图片\n")
    
    for idx, (img_path, framing_path) in enumerate(image_files):
        img_name = img_path.name
        print(f"\n[{idx+1}/{len(image_files)}] 处理: {img_name}")
        
        top10, best, gt_bbox, objects, complete_boxes = process_single_image(
            str(img_path), str(framing_path), checker, TOP_PERCENT
        )
        
        if top10 is None:
            print(f"   ❌ 处理失败")
            continue
        
        print(f"   📊 前十名美学得分: ", end="")
        for i, item in enumerate(top10):
            print(f"#{i+1}:{item['score']:.3f} ", end="")
        print()
        
        if best:
            print(f"   🏆 最佳美学得分: {best['score']:.4f}, IoU: {best['iou']:.4f}")
        
        output_path = os.path.join(output_dir, f"{Path(img_name).stem}_result.jpg")
        visualize_result(str(img_path), top10, best, gt_bbox, objects, output_path)
        print(f"   💾 保存: {output_path}")
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()