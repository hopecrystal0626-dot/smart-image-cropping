"""
画面平衡规则
原理：视觉重心是否在画面中心附近，避免失重感
"""

import cv2
import numpy as np
from typing import Tuple


def compute_balance_score(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """
    计算候选框的画面平衡度得分
    
    Args:
        image: 原始图像 (H, W, 3) BGR格式
        bbox: 候选框 (x, y, w, h)
    
    Returns:
        平衡度得分，范围 0~1，越高表示画面越平衡
    """
    x, y, w, h = bbox
    
    # 提取候选框区域
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.5
    
    # 转换为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 使用梯度作为视觉重量的度量（边缘越强，视觉重量越大）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    weight_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
    weight_map = weight_map / (np.mean(weight_map) + 1e-6)
    
    # 计算视觉重心
    total_weight = weight_map.sum()
    if total_weight < 1e-6:
        return 0.5
    
    y_coords, x_coords = np.indices(roi.shape[:2])
    center_x_weighted = (weight_map * x_coords).sum() / total_weight
    center_y_weighted = (weight_map * y_coords).sum() / total_weight
    
    # 理想重心在框中心
    ideal_center_x = w / 2
    ideal_center_y = h / 2
    
    # 计算偏移距离（归一化）
    offset_x = abs(center_x_weighted - ideal_center_x) / w
    offset_y = abs(center_y_weighted - ideal_center_y) / h
    offset = np.sqrt(offset_x ** 2 + offset_y ** 2)
    
    # 偏移越小得分越高
    score = 1 - min(offset, 1.0)
    
    return score


def compute_balance_score_batch(image: np.ndarray, 
                                 candidates: list) -> list:
    """
    批量计算画面平衡度得分
    
    Args:
        image: 原始图像
        candidates: 候选框列表 [(x,y,w,h), ...]
    
    Returns:
        得分列表
    """
    return [compute_balance_score(image, box) for box in candidates]