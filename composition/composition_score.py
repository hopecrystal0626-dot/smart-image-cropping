"""
构图规则综合评分模块
整合三分法、画面平衡、留白合理性三个维度
"""

import numpy as np
from typing import List, Tuple

# 导入三个子模块
from composition.thirds import compute_thirds_score_batch
from composition.balance import compute_balance_score_batch
from composition.whitespace import compute_whitespace_score_batch

# 类型别名
Bbox = Tuple[int, int, int, int]


class CompositionScorer:
    """
    构图规则综合评分器
    """
    
    def __init__(self, 
                 weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)):
        """
        初始化构图评分器
        
        Args:
            weights: (三分法权重, 平衡度权重, 留白权重)
        """
        self.weight_thirds = weights[0]
        self.weight_balance = weights[1]
        self.weight_whitespace = weights[2]
        # 保存权重元组，用于比较
        self.weights = weights
    
    def compute_scores(self, 
                       image: np.ndarray, 
                       candidates: List[Bbox]) -> List[float]:
        """
        计算每个候选框的构图综合得分
        
        Args:
            image: 原始图像 (H, W, 3)
            candidates: 候选框列表
            
        Returns:
            构图得分列表，范围 0~1
        """
        if not candidates:
            return []
        
        # 批量计算三个维度的得分
        thirds_scores = compute_thirds_score_batch(image, candidates)
        balance_scores = compute_balance_score_batch(image, candidates)
        whitespace_scores = compute_whitespace_score_batch(image, candidates)
        
        # 加权综合
        scores = []
        for t, b, w in zip(thirds_scores, balance_scores, whitespace_scores):
            total = (self.weight_thirds * t + 
                     self.weight_balance * b + 
                     self.weight_whitespace * w)
            scores.append(total)
        
        return scores
    
    def compute_single_score(self, image: np.ndarray, bbox: Bbox) -> dict:
        """
        计算单个候选框的详细得分（用于调试和理由生成）
        
        Returns:
            包含各维度得分的字典
        """
        from composition.thirds import compute_thirds_score
        from composition.balance import compute_balance_score
        from composition.whitespace import compute_whitespace_score
        
        thirds = compute_thirds_score(image, bbox)
        balance = compute_balance_score(image, bbox)
        whitespace = compute_whitespace_score(image, bbox)
        
        total = (self.weight_thirds * thirds + 
                 self.weight_balance * balance + 
                 self.weight_whitespace * whitespace)
        
        return {
            'thirds_score': thirds,
            'balance_score': balance,
            'whitespace_score': whitespace,
            'total_score': total
        }


# ========== 简化版函数接口 ==========

_scorer = None
_current_weights = None


def compute_composition_scores(image: np.ndarray, 
                                candidates: List[Bbox],
                                weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)) -> List[float]:
    """
    计算每个候选框的构图得分（批量接口）
    
    示例:
        scores = compute_composition_scores(img, candidates)
    
    Args:
        image: 原始图像
        candidates: 候选框列表
        weights: (三分法权重, 平衡度权重, 留白权重)
    
    Returns:
        构图得分列表，范围 0~1
    """
    global _scorer, _current_weights
    
    # 如果权重变化或首次调用，重新创建评分器
    if _scorer is None or _current_weights != weights:
        _scorer = CompositionScorer(weights)
        _current_weights = weights
    
    return _scorer.compute_scores(image, candidates)


def compute_composition_score_single(image: np.ndarray, bbox: Bbox) -> dict:
    """
    计算单个候选框的详细构图得分（用于调试）
    
    示例:
        detail = compute_composition_score_single(img, (100,80,300,300))
        print(detail['thirds_score'])
    """
    global _scorer
    if _scorer is None:
        _scorer = CompositionScorer()
    return _scorer.compute_single_score(image, bbox)