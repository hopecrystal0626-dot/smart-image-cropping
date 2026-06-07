"""
画面平衡规则（裁剪图版本）
原理：视觉重心是否在画面中心附近，避免失重感
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def compute_balance_score(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    计算裁剪图的画面平衡度得分（基于图片本身）
    
    Args:
        image: 裁剪后的图片 (H, W, 3) BGR格式
        bbox: 保留参数兼容性，实际不使用
    
    Returns:
        平衡度得分，范围 0~1
    """
    h, w = image.shape[:2]
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用梯度作为视觉重量
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    weight_map = np.sqrt(grad_x**2 + grad_y**2)
    weight_map = weight_map / (weight_map.max() + 1e-6)
    
    # 计算视觉重心
    total_weight = weight_map.sum()
    if total_weight < 1e-6:
        return 0.5
    
    y_coords, x_coords = np.indices((h, w))
    center_x = (weight_map * x_coords).sum() / total_weight
    center_y = (weight_map * y_coords).sum() / total_weight
    
    # 理想重心在图片中心
    ideal_center_x = w / 2
    ideal_center_y = h / 2
    
    # 计算偏移量（归一化）
    offset_x = abs(center_x - ideal_center_x) / w
    offset_y = abs(center_y - ideal_center_y) / h
    offset = np.sqrt(offset_x**2 + offset_y**2)
    
    # 偏移越小得分越高
    score = 1 - min(offset, 1.0)
    
    return max(0.0, min(1.0, score))


def compute_balance_score_batch(image: np.ndarray, 
                                 candidates: list = None) -> list:
    """
    批量计算平衡度得分
    """
    score = compute_balance_score(image)
    if candidates is None:
        return [score]
    return [score for _ in candidates]