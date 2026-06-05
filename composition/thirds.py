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
    

    center_x = x + w / 2
    center_y = y + h / 2
    
    
    # 计算三分线位置（相对于全图）
    third_w = img_w / 3
    third_h = img_h / 3
    
    points = [
        (third_w, third_h),
        (2 * third_w, third_h),
        (third_w, 2 * third_h),
        (2 * third_w, 2 * third_h)
    ]
    
  
    min_dist = min(
        np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2)
        for px, py in points
    )

    max_dist = np.sqrt(img_w ** 2 + img_h ** 2) / 2

    score = 1 - (min_dist / (max_dist + 1e-6))

    return float(np.clip(score, 0.0, 1.0))

def compute_thirds_score_batch(image, candidates):
    return [compute_thirds_score(image, b) for b in candidates]