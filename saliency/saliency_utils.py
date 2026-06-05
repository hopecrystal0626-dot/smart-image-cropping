"""
显著性工具函数：提供显著图获取、候选框评分、后处理、取景框提取、IoU计算、候选框筛选等
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from saliency.detector import get_default_detector
from crop.bbox_utils import BBox   # 依赖队友的 BBox 定义

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = get_default_detector()
    return _detector

def get_saliency_map(image: np.ndarray) -> np.ndarray:
    """输入BGR图像，返回显著图（float32，范围[0,1]）"""
    detector = _get_detector()
    return detector.detect(image)

def get_saliency_score(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """计算候选框内的平均显著性得分"""
    sal_map = get_saliency_map(image)
    x1, y1, x2, y2 = map(int, bbox)
    h, w = sal_map.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    roi = sal_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(roi.mean())

def postprocess_saliency(sal_map: np.ndarray,
                         blur_sigma: float = 1.0,
                         threshold: Optional[float] = None) -> np.ndarray:
    """后处理：高斯平滑 + 可选二值化"""
    if blur_sigma > 0:
        sal_map = cv2.GaussianBlur(sal_map, (0, 0), blur_sigma)
    if threshold is not None:
        sal_map = (sal_map > threshold).astype(np.float32)
    if sal_map.max() > 0:
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
    return sal_map

def extract_bbox_from_framing(original_img_path: str, framing_img_path: str) -> Optional[Tuple[int, int, int, int]]:
    """从原图和 _framing.jpg 中提取取景框坐标（模板匹配）"""
    original = cv2.imread(original_img_path)
    framing = cv2.imread(framing_img_path)
    if original is None or framing is None:
        return None
    result = cv2.matchTemplate(original, framing, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > 0.8:
        x, y = max_loc
        h, w = framing.shape[:2]
        return (x, y, x + w, y + h)
    return None

# ========== 新增：IoU 计算（健壮版本）==========
def compute_iou(box1: BBox, box2: BBox) -> float:
    """计算两个 BBox 的 IoU（使用队友的 BBox 类）"""
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

# ========== 新增：基于中心偏好的评分函数 ==========
def score_by_center_bias(sal_map: np.ndarray, bbox: BBox, center_bias: float = 0.3) -> float:
    """
    结合显著图均值和显著图质心与框中心的距离进行评分
    center_bias 越大，越强调质心距离
    """
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

# ========== 新增：根据得分筛选前百分比候选框并分段 ==========
def filter_top_bboxes_by_percentile(sal_map: np.ndarray,
                                    candidates: List[BBox],
                                    top_percent: float = 0.3,
                                    num_segments: int = 3):
    """
    筛选出得分前 top_percent 的候选框，并返回分段索引。
    返回: (top_bboxes, top_scores, segment_indices)
    segment_indices: 列表，每个框对应的分段号（0,1,2...）
    """
    scored = [(bbox, score_by_center_bias(sal_map, bbox)) for bbox in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(scored) * top_percent))
    top_bboxes = [bbox for bbox, _ in scored[:k]]
    top_scores = [score for _, score in scored[:k]]
    seg_len = k // num_segments
    segment_indices = []
    for i in range(k):
        if i < seg_len:
            seg = 0
        elif i < 2 * seg_len:
            seg = 1
        else:
            seg = 2
        segment_indices.append(seg)
    return top_bboxes, top_scores, segment_indices