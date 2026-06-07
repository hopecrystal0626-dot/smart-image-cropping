"""
三分法构图规则（裁剪图版本）
原理：在裁剪后的图片上，重要元素是否位于框内的三分线交点
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def compute_thirds_score(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    计算裁剪图的三分法得分（基于图片本身）
    
    Args:
        image: 裁剪后的图片 (H, W, 3) BGR格式
        bbox: 保留参数兼容性，实际不使用
    
    Returns:
        三分法得分，范围 0~1，越高表示图片越符合三分法构图
    """
    h, w = image.shape[:2]
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用边缘检测作为"重要元素"的代理
    edges = cv2.Canny(gray, 50, 150)
    
    # 计算边缘的加权中心（视觉重心）
    y_coords, x_coords = np.indices((h, w))
    total_weight = edges.sum()
    
    if total_weight < 1e-6:
        return 0.5
    
    center_x = (edges * x_coords).sum() / total_weight
    center_y = (edges * y_coords).sum() / total_weight
    
    # 图片内的三分线位置
    third_w = w / 3
    third_h = h / 3
    
    # 四个三分线交点
    intersection_points = [
        (third_w, third_h),       # 左上交点
        (2 * third_w, third_h),   # 右上交点
        (third_w, 2 * third_h),   # 左下交点
        (2 * third_w, 2 * third_h) # 右下交点
    ]
    
    # 计算视觉重心到最近三分线交点的距离
    min_dist = float('inf')
    for px, py in intersection_points:
        dist = np.sqrt((center_x - px)**2 + (center_y - py)**2)
        min_dist = min(min_dist, dist)
    
    # 归一化距离
    max_dist = np.sqrt(w**2 + h**2) / 2
    normalized_dist = min(min_dist / max_dist, 1.0)
    score = 1 - normalized_dist
    
    return max(0.0, min(1.0, score))


def compute_thirds_score_batch(image: np.ndarray, 
                                candidates: list = None) -> list:
    """
    批量计算三分法得分
    """
    score = compute_thirds_score(image)
    if candidates is None:
        return [score]
    return [score for _ in candidates]