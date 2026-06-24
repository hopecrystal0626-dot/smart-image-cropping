"""
显著性工具函数：提供显著图获取、候选框评分、后处理、取景框提取、IoU计算、候选框筛选以及主体中心/边界框提取等
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from saliency.detector import get_default_detector
from crop.bbox_utils import BBox

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

def compute_iou(box1: BBox, box2: BBox) -> float:
    """计算两个 BBox 的 IoU"""
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

def score_by_center_bias(sal_map: np.ndarray, bbox: BBox, center_bias: float = 0.3) -> float:
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

def filter_top_bboxes_by_percentile(sal_map: np.ndarray,
                                    candidates: List[BBox],
                                    top_percent: float = 0.3,
                                    num_segments: int = 3):
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

def get_subject_center(sal_map: np.ndarray):
    """从显著图中提取最大连通域的中心"""
    h, w = sal_map.shape
    thresh = np.percentile(sal_map, 70)
    binary = (sal_map > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return w / 2, h / 2
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2
    return cx, cy

# ========== 新增：提取最大连通域边界框 ==========
def get_subject_bbox(sal_map: np.ndarray, img_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """
    从显著图中提取最大连通域的外接矩形
    返回 (x1, y1, x2, y2) 或 None
    """
    h, w = img_shape[:2]
    thresh = np.percentile(sal_map, 70)
    binary = (sal_map > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w2, h2 = cv2.boundingRect(largest)
    return (x, y, x + w2, y + h2)