"""
留白合理性规则
原理：主体区域占整幅图像的面积比例是否合适
       理想的主体面积占比约为 30% ~ 60%
"""

import cv2
import numpy as np
from typing import Tuple


def compute_whitespace_score(image: np.ndarray, bbox: Tuple[int, int, int, int], debug: bool = False) -> float:
    """
    计算候选框的留白合理性得分
    
    Args:
        image: 原始图像 (H, W, 3) BGR格式
        bbox: 候选框 (x, y, w, h)
        debug: 是否打印调试信息
    
    Returns:
        留白得分，范围 0~1，越高表示留白越合理
    """
    x, y, w, h = bbox
    
    # 提取候选框区域
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.5
    
    if debug:
        print(f"\n[留白调试] 框尺寸: {w}x{h}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # ========== 方法1：Otsu 自适应阈值 ==========
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    subject_area_otsu = (binary > 0).sum() / binary.size
    
    if debug:
        print(f"  Otsu主体占比: {subject_area_otsu:.4f}")
    
    # ========== 方法2：边缘密度 ==========
    edges = cv2.Canny(gray, 50, 150)
    edge_density = (edges > 0).sum() / edges.size
    
    if debug:
        print(f"  边缘密度: {edge_density:.4f}")
    
    # ========== 方法3：梯度幅值（更精细的边缘检测）==========
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-6)
    grad_density = (grad_magnitude > 0.1).sum() / grad_magnitude.size
    
    if debug:
        print(f"  梯度密度: {grad_density:.4f}")
    
    # ========== 方法4：局部方差（纹理丰富度）==========
    # 计算局部方差，方差大的区域通常是主体
    kernel_size = 15
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean**2
    var_threshold = local_var.mean() + local_var.std()
    texture_mask = (local_var > var_threshold).astype(np.uint8)
    texture_density = texture_mask.sum() / texture_mask.size
    
    if debug:
        print(f"  纹理密度: {texture_density:.4f}")
    
    # ========== 综合主体占比 ==========
    # 使用多种方法的加权平均，更鲁棒
    subject_ratio = (
        0.3 * subject_area_otsu + 
        0.3 * edge_density * 2 + 
        0.2 * grad_density + 
        0.2 * texture_density
    )
    subject_ratio = min(subject_ratio, 1.0)
    
    if debug:
        print(f"  综合主体占比: {subject_ratio:.4f}")
    
    # ========== 理想范围（放宽，更符合真实图像）==========
    # 对于风景/建筑：主体占 20%~50% 比较好
    # 对于人像/特写：主体占 40%~70% 比较好
    # 使用默认范围 25%~60%
    ideal_min = 0.25
    ideal_max = 0.60
    
    if subject_ratio < ideal_min:
        # 主体太小，留白过多
        score = subject_ratio / ideal_min
        if debug:
            print(f"  主体太小 → 得分: {score:.4f}")
    elif subject_ratio > ideal_max:
        # 主体太大，画面过满
        score = 1 - (subject_ratio - ideal_max) / (1 - ideal_max)
        if debug:
            print(f"  主体太大 → 得分: {score:.4f}")
    else:
        # 理想范围内
        score = 1.0
        if debug:
            print(f"  主体在理想范围 → 得分: 1.0")
    
    # 确保得分在 [0, 1] 范围内
    score = max(0.0, min(1.0, score))
    
    return score


def compute_whitespace_score_batch(image: np.ndarray, 
                                    candidates: list,
                                    debug: bool = False) -> list:
    """
    批量计算留白合理性得分
    
    Args:
        image: 原始图像
        candidates: 候选框列表 [(x,y,w,h), ...]
        debug: 是否打印调试信息（只打印第一个）
    
    Returns:
        得分列表
    """
    scores = []
    for i, box in enumerate(candidates):
        if debug and i == 0:
            score = compute_whitespace_score(image, box, debug=True)
        else:
            score = compute_whitespace_score(image, box, debug=False)
        scores.append(score)
    return scores