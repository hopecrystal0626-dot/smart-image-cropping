"""
留白合理性规则（裁剪图版本）
原理：主体区域占整幅图像的面积比例是否合适
       理想的主体面积占比约为 25% ~ 60%
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def compute_whitespace_score(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    计算裁剪图的留白合理性得分（基于图片本身）
    
    Args:
        image: 裁剪后的图片 (H, W, 3) BGR格式
        bbox: 保留参数兼容性，实际不使用
    
    Returns:
        留白得分，范围 0~1
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 方法1：Otsu自适应阈值检测主体区域
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    subject_area_otsu = (binary > 0).sum() / binary.size
    
    # 方法2：边缘密度作为补充
    edges = cv2.Canny(gray, 50, 150)
    edge_density = (edges > 0).sum() / edges.size
    
    # 方法3：梯度密度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-6)
    grad_density = (grad_magnitude > 0.1).sum() / grad_magnitude.size
    
    # 综合主体面积估计
    subject_ratio = (
        0.4 * subject_area_otsu +
        0.3 * edge_density * 2 +
        0.3 * grad_density
    )
    subject_ratio = min(subject_ratio, 1.0)
    
    # 理想主体占比范围：25% ~ 60%
    ideal_min = 0.25
    ideal_max = 0.60
    
    if subject_ratio < ideal_min:
        # 主体太小，留白过多
        score = subject_ratio / ideal_min
    elif subject_ratio > ideal_max:
        # 主体太大，画面过满
        score = 1 - (subject_ratio - ideal_max) / (1 - ideal_max)
    else:
        # 理想范围内
        score = 1.0
    
    return max(0.0, min(1.0, score))


def compute_whitespace_score_batch(image: np.ndarray, 
                                    candidates: list = None) -> list:
    """
    批量计算留白合理性得分
    """
    score = compute_whitespace_score(image)
    if candidates is None:
        return [score]
    return [score for _ in candidates]