"""
显著性工具函数：提供显著图获取、候选框评分、后处理以及从 _framing.jpg 提取取景框等功能
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from saliency.detector import get_default_detector

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