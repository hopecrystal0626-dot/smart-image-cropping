"""
三分法构图规则
原理：将画面用两条竖线和两条横线分成9等份，重要元素放在交点处
"""

import numpy as np
from typing import Tuple


def compute_thirds_score(image: np.ndarray, bbox: Tuple[int, int, int, int], debug: bool = False) -> float:
    """
    计算候选框的三分法得分
    
    Args:
        image: 原始图像 (H, W, 3) BGR格式
        bbox: 候选框 (x, y, w, h)
        debug: 是否打印调试信息
    
    Returns:
        三分法得分，范围 0~1，越高表示越符合三分法构图
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    
    if debug:
        print(f"\n[三分法调试] 图像尺寸: {img_w}x{img_h}")
        print(f"[三分法调试] 候选框: ({x},{y},{w},{h})")
    
    # 计算候选框的中心点（相对于全图）
    center_x = x + w / 2
    center_y = y + h / 2
    
    if debug:
        print(f"[三分法调试] 框中心: ({center_x:.1f}, {center_y:.1f})")
    
    # 计算三分线位置（相对于全图）
    third_w = img_w / 3
    third_h = img_h / 3
    
    # 四个三分线交点
    intersection_points = [
        (third_w, third_h),       # 左上交点
        (2 * third_w, third_h),   # 右上交点
        (third_w, 2 * third_h),   # 左下交点
        (2 * third_w, 2 * third_h) # 右下交点
    ]
    
    if debug:
        print(f"[三分法调试] 三分线交点: {[(int(p[0]), int(p[1])) for p in intersection_points]}")
    
    # 计算中心点到最近三分线交点的距离
    min_dist = float('inf')
    closest_point = None
    for px, py in intersection_points:
        dist = np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2)
        if debug:
            print(f"[三分法调试] 到交点({int(px)},{int(py)})的距离: {dist:.2f}")
        if dist < min_dist:
            min_dist = dist
            closest_point = (px, py)
    
    if debug:
        print(f"[三分法调试] 最近交点: ({int(closest_point[0])},{int(closest_point[1])})")
        print(f"[三分法调试] 最近距离: {min_dist:.2f}")
    
    # 最大可能距离（图像对角线的一半，因为交点分布在对角线附近）
    max_dist = np.sqrt(img_w ** 2 + img_h ** 2) / 2
    
    if debug:
        print(f"[三分法调试] 最大可能距离: {max_dist:.2f}")
    
    # 归一化得分，距离越小得分越高
    # 使用 sigmoid 或线性映射，避免得分过于集中
    normalized_dist = min(min_dist / max_dist, 1.0)
    score = 1 - normalized_dist
    
    # 应用非线性变换，让中等距离的得分差异更明显
    # score = score ** 0.8  # 可选，让得分分布更均匀
    
    if debug:
        print(f"[三分法调试] 归一化距离: {normalized_dist:.4f}")
        print(f"[三分法调试] 三分法得分: {score:.4f}")
    
    # 防止数值误差
    score = max(0.0, min(1.0, score))
    
    return score


def compute_thirds_score_batch(image: np.ndarray, 
                                candidates: list,
                                debug: bool = False) -> list:
    """
    批量计算三分法得分
    
    Args:
        image: 原始图像
        candidates: 候选框列表 [(x,y,w,h), ...]
        debug: 是否打印调试信息（只打印第一个框）
    
    Returns:
        得分列表
    """
    scores = []
    for i, box in enumerate(candidates):
        # 只对第一个框打印调试信息
        if debug and i == 0:
            score = compute_thirds_score(image, box, debug=True)
        else:
            score = compute_thirds_score(image, box, debug=False)
        scores.append(score)
    return scores