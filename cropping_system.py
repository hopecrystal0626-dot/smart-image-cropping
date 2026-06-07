"""
智能取景系统 - 统一接口
提供：给定一张图片，返回前10个最优候选框的坐标
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from composition.composition_score import compute_composition_score_single
from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox
from experiments.test_saliency import (
    FTDetector,
    filter_top_bboxes_by_percentile,
)


class SmartCropping:
    """智能取景系统主类"""
    
    def __init__(self, yolo_model_path="yolov8n.pt"):
        self.yolo = YOLO(yolo_model_path)
        print("✅ 智能取景系统初始化完成")
    
    def get_top10_crops(self, image_path, top_k=10):
        """
        输入图片路径，返回前K个最优候选框
        
        Args:
            image_path: 图片路径
            top_k: 返回前K个候选框（默认10）
        
        Returns:
            list: 每个元素包含 {'bbox': BBox对象, 'score': 美学得分, 'area_ratio': 面积占比}
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = img.shape[:2]
        
        # 1. YOLO 物体检测
        all_objects = self._detect_objects(img)
        
        # 2. 生成候选框
        all_candidates = generate_candidates(w, h)
        
        # 3. 尺寸筛选
        MIN_AREA_RATIO = 0.10
        filtered_by_size = []
        for bbox in all_candidates:
            area_ratio = (bbox.width * bbox.height) / (w * h)
            if area_ratio >= MIN_AREA_RATIO:
                filtered_by_size.append(bbox)
        
        # 4. 显著性筛选
        detector = FTDetector()
        sal_map = detector.detect(img)
        result = filter_top_bboxes_by_percentile(sal_map, filtered_by_size, 0.3)
        top_bboxes = result[0]
        
        # 5. 完整性检测
        complete_boxes = []
        for bbox in top_bboxes:
            is_complete = self._check_completeness(bbox, all_objects)
            if len(all_objects) == 0 or is_complete:
                complete_boxes.append(bbox)
        
        if len(complete_boxes) == 0:
            complete_boxes = top_bboxes
        
        # 6. 美学评分
        scored_boxes = []
        for bbox in complete_boxes:
            cropped = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            if cropped.size == 0:
                continue
            
            score_detail = compute_composition_score_single(
                cropped, 
                bbox=(bbox.x1, bbox.y1, bbox.width, bbox.height)
            )
            
            scored_boxes.append({
                'bbox': bbox,
                'score': score_detail['total_score'],
                'area_ratio': (bbox.width * bbox.height) / (w * h)
            })
        
        scored_boxes.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_boxes[:top_k]
    
    def _detect_objects(self, img):
        """YOLO 物体检测"""
        results = self.yolo(img)
        objects = []
        MIN_CONF = 0.3
        
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.conf[0] < MIN_CONF:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                objects.append({
                    'bbox': (x1, y1, x2, y2),
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        objects.sort(key=lambda x: x['area'], reverse=True)
        return objects
    
    def _check_completeness(self, candidate_bbox, all_objects, threshold=0.85):
        """检查候选框是否完整包含物体"""
        if len(all_objects) == 0:
            return True
        
        cx1, cy1, cx2, cy2 = candidate_bbox.x1, candidate_bbox.y1, candidate_bbox.x2, candidate_bbox.y2
        
        for obj in all_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            ix1 = max(cx1, ox1)
            iy1 = max(cy1, oy1)
            ix2 = min(cx2, ox2)
            iy2 = min(cy2, oy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                o_area = obj['area']
                completeness = inter_area / o_area
                if completeness >= threshold:
                    return True
        
        return False


# 使用示例
if __name__ == "__main__":
    # 初始化系统
    cropper = SmartCropping()
    
    # 获取前10个候选框
    results = cropper.get_top10_crops("data/testA/A15.jpg", top_k=10)
    
    print("前10名候选框:")
    for i, r in enumerate(results):
        bbox = r['bbox']
        print(f"  #{i+1}: 位置=({bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2}), 得分={r['score']:.4f}, 面积={r['area_ratio']:.0%}")