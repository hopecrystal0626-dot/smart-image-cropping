"""
单张图片智能取景 - 显示前10个候选框
输入任意图片 → 同学A（滑动窗口）→ 同学B（显著性筛选）→ YOLO完整性检测 → 美学评分 → 显示前10个框
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
    compute_iou
)


# ============================================================
# 配置
# ============================================================
YOLO_MODEL = str(project_root / "yolov8n.pt")
MIN_CONF = 0.3                     # YOLO 检测置信度
COMPLETENESS_THRESHOLD = 0.85      # 物体完整度阈值
TOP_PERCENT = 0.3                  # 显著性筛选保留比例

# 美学评分权重
THIRDS_WEIGHT = 0.25
BALANCE_WEIGHT = 0.15
WHITESPACE_WEIGHT = 0.15
CENTER_WEIGHT = 0.45
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
        
        objects.sort(key=lambda x: x['area'], reverse=True)
        return objects
    
    def check_candidate_completeness(self, candidate_bbox, all_objects):
        """检查候选框是否完整包含检测到的物体"""
        cx1, cy1, cx2, cy2 = candidate_bbox.x1, candidate_bbox.y1, candidate_bbox.x2, candidate_bbox.y2
        
        best_match = None
        best_completeness = 0
        
        for obj in all_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            o_area = obj['area']
            
            ix1 = max(cx1, ox1)
            iy1 = max(cy1, oy1)
            ix2 = min(cx2, ox2)
            iy2 = min(cy2, oy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                completeness = inter_area / o_area
                
                if completeness > best_completeness:
                    best_completeness = completeness
                    best_match = obj
        
        is_complete = best_completeness >= COMPLETENESS_THRESHOLD
        return is_complete, best_completeness, best_match


def process_single_image(img, checker):
    """处理单张图片，返回所有候选框的评分"""
    h, w = img.shape[:2]
    
    # 步骤1：检测原图中所有物体
    all_objects = checker.detect_all_objects(img)
    if len(all_objects) > 0:
        print(f"   检测到 {len(all_objects)} 个物体: ", end="")
        for obj in all_objects[:5]:
            print(f"{obj['class']} ", end="")
        print()
    else:
        print(f"   未检测到物体")
    
    # 步骤2：生成候选框（同学A）
    all_candidates = generate_candidates(w, h)
    print(f"   候选框总数: {len(all_candidates)}")
    
    # 步骤3：尺寸筛选
    MIN_AREA_RATIO = 0.10  # 放宽到10%
    filtered_by_size = []
    for bbox in all_candidates:
        area_ratio = (bbox.width * bbox.height) / (w * h)
        if area_ratio >= MIN_AREA_RATIO:
            filtered_by_size.append(bbox)
    print(f"   尺寸筛选后(≥{MIN_AREA_RATIO:.0%}): {len(filtered_by_size)} 个")
    
    # 步骤4：显著性筛选（同学B）
    detector = FTDetector()
    sal_map = detector.detect(img)
    result = filter_top_bboxes_by_percentile(sal_map, filtered_by_size, TOP_PERCENT)
    top_bboxes = result[0]
    print(f"   显著性筛选后(前{int(TOP_PERCENT*100)}%): {len(top_bboxes)} 个")
    
    if len(top_bboxes) == 0:
        return None, None, None
    
    # 步骤5：完整性检测
    complete_boxes = []
    for bbox in top_bboxes:
        is_complete, _, _ = checker.check_candidate_completeness(bbox, all_objects)
        if len(all_objects) == 0 or is_complete:
            complete_boxes.append(bbox)
    
    print(f"   完整性筛选后: {len(complete_boxes)} 个")
    
    if len(complete_boxes) == 0:
        complete_boxes = top_bboxes
        print(f"   ⚠️ 没有完整框，使用原始候选框")
    
    # 步骤6：美学评分
    scored_boxes = []
    for bbox in complete_boxes:
        cropped = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if cropped.size == 0:
            continue
        
        score_detail = compute_composition_score_single(cropped, bbox=(bbox.x1, bbox.y1, bbox.width, bbox.height))
        comp_score = score_detail['total_score']
        
        scored_boxes.append({
            'bbox': bbox,
            'score': comp_score,
            'area_ratio': (bbox.width * bbox.height) / (w * h)
        })
    
    if len(scored_boxes) == 0:
        return None, None, None
    
    scored_boxes.sort(key=lambda x: x['score'], reverse=True)
    top10 = scored_boxes[:10]
    
    return top10, scored_boxes, all_objects


def visualize_top10(img, top10, all_boxes, objects, output_path):
    """可视化前10个候选框（不同颜色）"""
    canvas = img.copy()
    h, w = img.shape[:2]
    
    # 颜色列表（前10名不同颜色）
    colors = [
        (255, 0, 255),   # 第1名：粉色
        (0, 255, 0),     # 第2名：绿色
        (255, 255, 0),   # 第3名：青色
        (0, 165, 255),   # 第4名：橙色
        (255, 0, 0),     # 第5名：蓝色
        (128, 0, 128),   # 第6名：紫色
        (0, 128, 128),   # 第7名：深青
        (128, 128, 0),   # 第8名：橄榄
        (255, 128, 0),   # 第9名：橙黄
        (0, 128, 255)    # 第10名：天蓝
    ]
    
    # 检测到的物体（蓝色细线）
    for obj in objects[:5]:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(canvas, obj['class'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 画前10名框
    for i, item in enumerate(top10):
        bbox = item['bbox']
        color = colors[i % len(colors)]
        # 第一名粗线，其他细线
        thickness = 3 if i == 0 else 2
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
        
        # 显示排名和得分
        label = f"#{i+1}: {item['score']:.3f}"
        cv2.putText(canvas, label, (bbox.x1, bbox.y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    # 添加图例
    legend_y = 30
    cv2.putText(canvas, "Top 10 Candidates:", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    for i, color in enumerate(colors[:5]):
        cv2.rectangle(canvas, (10, legend_y + 10 + i*25), (30, legend_y + 30 + i*25), color, -1)
        cv2.putText(canvas, f"#{i+1}", (35, legend_y + 25 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.imwrite(output_path, canvas)


def save_each_crop(img, top10, output_dir, original_name):
    """单独保存每个候选框的裁剪结果"""
    crop_dir = output_dir / f"{original_name}_crops"
    crop_dir.mkdir(exist_ok=True)
    
    for i, item in enumerate(top10):
        bbox = item['bbox']
        cropped = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if cropped.size > 0:
            crop_path = crop_dir / f"rank_{i+1}_score_{item['score']:.3f}.jpg"
            cv2.imwrite(str(crop_path), cropped)
            print(f"      保存: {crop_path}")


def main():
    print("="*60)
    print("单张图片智能取景系统 - 显示前10个候选框")
    print("流程: 候选框 → 显著性筛选 → 完整性检测 → 美学评分")
    print("="*60)
    
    # 输入图片路径
    img_path = input("\n请输入图片路径: ").strip()
    img_path = img_path.strip('"').strip("'")
    
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return
    
    # 输出目录
    output_dir = Path(img_path).parent / "crop_top10_result"
    output_dir.mkdir(exist_ok=True)
    original_name = Path(img_path).stem
    
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片: {img_path}")
        return
    
    print(f"\n图片尺寸: {img.shape[1]} x {img.shape[0]}")
    
    # 初始化检测器
    checker = CompletenessChecker()
    
    # 处理图片
    print("\n正在处理...")
    top10, all_boxes, objects = process_single_image(img, checker)
    
    if top10 is None:
        print("❌ 处理失败")
        return
    
    # 输出前10名信息
    print(f"\n📊 前10名美学得分:")
    for i, item in enumerate(top10):
        print(f"   #{i+1}: 得分={item['score']:.4f}, 面积占比={item['area_ratio']:.0%}")
    
    # 保存每个裁剪结果
    save_each_crop(img, top10, output_dir, original_name)
    
    # 可视化前10名框
    visualize_path = output_dir / f"{original_name}_top10_visualize.jpg"
    visualize_top10(img, top10, all_boxes, objects, str(visualize_path))
    
    # 保存最佳裁剪
    best = top10[0]
    best_cropped = img[best['bbox'].y1:best['bbox'].y2, best['bbox'].x1:best['bbox'].x2]
    best_path = output_dir / f"{original_name}_best_crop.jpg"
    cv2.imwrite(str(best_path), best_cropped)
    
    print(f"\n✅ 处理完成！")
    print(f"   最佳框位置: ({best['bbox'].x1}, {best['bbox'].y1}, {best['bbox'].x2}, {best['bbox'].y2})")
    print(f"   最佳框美学得分: {best['score']:.4f}")
    print(f"   最佳裁剪保存: {best_path}")
    print(f"   前10名裁剪保存在: {output_dir}/{original_name}_crops/")
    print(f"   可视化结果保存: {visualize_path}")
    
    # 显示最佳裁剪结果
    cv2.imshow("Best Crop", cv2.resize(best_cropped, (600, 400)))
    print("\n按任意键关闭图片窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()