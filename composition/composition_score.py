"""
构图规则综合评分模块（最终优化版）
基于老师框分析结果优化的权重
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CompositionScorer:
    """
    构图规则综合评分器（最终优化版）
    """
    
    def __init__(self, 
                 weights: Tuple[float, float, float, float, float, float] = 
                 (0.25, 0.25, 0.25, 0.10, 0.10, 0.05)):
        """
        初始化构图评分器
        
        Args:
            weights: (三分法, 平衡度, 留白, 中心偏好, 尺寸评分, 边界惩罚)
        """
        self.w_thirds = weights[0]
        self.w_balance = weights[1]
        self.w_whitespace = weights[2]
        self.w_center = weights[3]
        self.w_size = weights[4]
        self.w_boundary = weights[5]
    
    def compute_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None, original_shape: Tuple[int, int] = None) -> float:
        """计算综合构图得分"""
        h, w = image.shape[:2]
        
        thirds_score = self._compute_thirds_score(image)
        balance_score = self._compute_balance_score(image)
        whitespace_score = self._compute_whitespace_score(image)
        
        center_score = 0.5
        size_score = 0.5
        boundary_score = 0.5
        
        if bbox is not None:
            if original_shape is not None:
                img_h, img_w = original_shape
            else:
                img_h, img_w = h, w
            
            center_score = self._compute_center_score(bbox, (img_h, img_w))
            size_score = self._compute_size_score(bbox, (img_h, img_w))
            boundary_score = self._compute_boundary_score(bbox, (img_h, img_w))
        
        total = (self.w_thirds * thirds_score +
                 self.w_balance * balance_score +
                 self.w_whitespace * whitespace_score +
                 self.w_center * center_score +
                 self.w_size * size_score +
                 self.w_boundary * boundary_score)
        
        return max(0.0, min(1.0, total))
    
    def compute_single_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None, original_shape: Tuple[int, int] = None) -> dict:
        """返回详细得分"""
        h, w = image.shape[:2]
        
        thirds = self._compute_thirds_score(image)
        balance = self._compute_balance_score(image)
        whitespace = self._compute_whitespace_score(image)
        
        center = 0.5
        size = 0.5
        boundary = 0.5
        
        if bbox is not None:
            if original_shape is not None:
                img_h, img_w = original_shape
            else:
                img_h, img_w = h, w
            
            center = self._compute_center_score(bbox, (img_h, img_w))
            size = self._compute_size_score(bbox, (img_h, img_w))
            boundary = self._compute_boundary_score(bbox, (img_h, img_w))
        
        total = (self.w_thirds * thirds +
                 self.w_balance * balance +
                 self.w_whitespace * whitespace +
                 self.w_center * center +
                 self.w_size * size +
                 self.w_boundary * boundary)
        
        return {
            'thirds_score': max(0.0, min(1.0, thirds)),
            'balance_score': max(0.0, min(1.0, balance)),
            'whitespace_score': max(0.0, min(1.0, whitespace)),
            'center_score': max(0.0, min(1.0, center)),
            'size_score': max(0.0, min(1.0, size)),
            'boundary_score': max(0.0, min(1.0, boundary)),
            'total_score': max(0.0, min(1.0, total))
        }
    
    # ========== 评分规则方法 ==========
    
    def _compute_thirds_score(self, image: np.ndarray) -> float:
        """三分法得分"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        y_coords, x_coords = np.indices((h, w))
        total_weight = edges.sum()
        
        if total_weight < 1e-6:
            return 0.5
        
        center_x = (edges * x_coords).sum() / total_weight
        center_y = (edges * y_coords).sum() / total_weight
        
        third_w = w / 3
        third_h = h / 3
        points = [(third_w, third_h), (2*third_w, third_h), 
                  (third_w, 2*third_h), (2*third_w, 2*third_h)]
        
        min_dist = min(np.sqrt((center_x - px)**2 + (center_y - py)**2) for px, py in points)
        max_dist = np.sqrt(w**2 + h**2) / 2
        score = 1 - min(min_dist / max_dist, 1.0)
        
        return max(0.0, min(1.0, score))
    
    def _compute_balance_score(self, image: np.ndarray) -> float:
        """平衡度得分"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        weight = np.sqrt(grad_x**2 + grad_y**2)
        weight = weight / (weight.max() + 1e-6)
        
        total = weight.sum()
        if total < 1e-6:
            return 0.5
        
        y, x = np.indices((h, w))
        cx = (weight * x).sum() / total
        cy = (weight * y).sum() / total
        
        offset = np.sqrt(((cx - w/2)/w)**2 + ((cy - h/2)/h)**2)
        score = 1 - min(offset, 1.0)
        
        return max(0.0, min(1.0, score))
    
    def _compute_whitespace_score(self, image: np.ndarray) -> float:
        """留白得分"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, 50, 150)
        
        subject_ratio = max((binary > 0).sum() / binary.size, 
                           (edges > 0).sum() / edges.size * 2)
        subject_ratio = min(subject_ratio, 1.0)
        
        ideal_min = 0.15
        ideal_max = 0.70
        
        if subject_ratio < ideal_min:
            score = subject_ratio / ideal_min
        elif subject_ratio > ideal_max:
            score = 1 - (subject_ratio - ideal_max) / (1 - ideal_max)
        else:
            score = 1.0
        
        return max(0.0, min(1.0, score))
    
    def _compute_center_score(self, bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> float:
        """中心偏好：框中心离图像中心越近得分越高"""
        x, y, w, h = bbox
        img_h, img_w = img_shape
        
        box_cx = x + w / 2
        box_cy = y + h / 2
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        dist = np.sqrt((box_cx - img_cx)**2 + (box_cy - img_cy)**2)
        max_dist = np.sqrt(img_w**2 + img_h**2) / 2
        
        score = 1 - min(dist / max_dist, 1.0)
        return max(0.0, min(1.0, score))
    
    def _compute_size_score(self, bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> float:
        """尺寸评分：框面积占比在合理范围内得分高"""
        x, y, w, h = bbox
        img_h, img_w = img_shape
        
        area_ratio = (w * h) / (img_w * img_h)
        
        ideal_min = 0.15
        ideal_max = 0.80
        
        if area_ratio < ideal_min:
            score = area_ratio / ideal_min
        elif area_ratio > ideal_max:
            score = 1 - (area_ratio - ideal_max) / (1 - ideal_max)
        else:
            score = 1.0
        
        return max(0.0, min(1.0, score))
    
    def _compute_boundary_score(self, bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> float:
        """边界惩罚：框离图像边界太近会扣分"""
        x, y, w, h = bbox
        img_h, img_w = img_shape
        
        dist_left = x / img_w
        dist_right = (img_w - (x + w)) / img_w
        dist_top = y / img_h
        dist_bottom = (img_h - (y + h)) / img_h
        
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        
        threshold = 0.05
        if min_dist < threshold:
            score = min_dist / threshold
        else:
            score = 1.0
        
        return max(0.0, min(1.0, score))


# ========== 简化版函数接口 ==========

_scorer = None


def compute_composition_score(image: np.ndarray,
                               bbox: Tuple[int, int, int, int] = None,
                               original_shape: Tuple[int, int] = None,
                               weights: Tuple[float, ...] = None) -> float:
    """计算裁剪图的综合构图得分"""
    global _scorer
    if _scorer is None or weights is not None:
        if weights is None:
            weights = (0.25, 0.25, 0.25, 0.10, 0.10, 0.05)
        _scorer = CompositionScorer(weights)
    
    return _scorer.compute_score(image, bbox, original_shape)


def compute_composition_score_single(image: np.ndarray, 
                                      bbox: Tuple[int, int, int, int] = None,
                                      original_shape: Tuple[int, int] = None) -> dict:
    """计算详细得分"""
    global _scorer
    if _scorer is None:
        _scorer = CompositionScorer()
    
    thirds = _scorer._compute_thirds_score(image)
    balance = _scorer._compute_balance_score(image)
    whitespace = _scorer._compute_whitespace_score(image)
    
    center = 0.5
    size = 0.5
    boundary = 0.5
    
    if bbox is not None:
        if original_shape is not None:
            img_h, img_w = original_shape
        else:
            img_h, img_w = image.shape[:2]
        
        center = _scorer._compute_center_score(bbox, (img_h, img_w))
        size = _scorer._compute_size_score(bbox, (img_h, img_w))
        boundary = _scorer._compute_boundary_score(bbox, (img_h, img_w))
    
    total = (_scorer.w_thirds * thirds +
             _scorer.w_balance * balance +
             _scorer.w_whitespace * whitespace +
             _scorer.w_center * center +
             _scorer.w_size * size +
             _scorer.w_boundary * boundary)
    
    return {
        'thirds_score': max(0.0, min(1.0, thirds)),
        'balance_score': max(0.0, min(1.0, balance)),
        'whitespace_score': max(0.0, min(1.0, whitespace)),
        'center_score': max(0.0, min(1.0, center)),
        'size_score': max(0.0, min(1.0, size)),
        'boundary_score': max(0.0, min(1.0, boundary)),
        'total_score': max(0.0, min(1.0, total))
    }