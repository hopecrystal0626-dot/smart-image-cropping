"""
方案3：纯美学评分 - 不判断人物，只判断画面美不美
专注：光线、色彩、美感（删除清晰度，降低负向权重）
"""

import numpy as np
import torch
from typing import Dict, Optional, List
from PIL import Image

from clip_score.clip_model import get_clip_model


class CLIPScorer:
    """纯美学评分器（删除清晰度，降低负向权重）"""
    
    def __init__(self, mode: str = "balanced"):
        self.clip = get_clip_model()
        self._get_image_features = self.clip._get_image_features
        
        # ========== 正向维度（只关注美学，删除清晰度）==========
        self.dimensions = {
            "lighting": {
                "positive": [
                    "good lighting",
                    "well exposed",
                    "bright vibrant lighting",
                    "well lit bright image",
                    "sunny bright scene",
                    "soft natural light",
                    "golden hour warm light",
                    "pleasant illumination"
                ],
                "weight": 0.45
            },
            "color": {
                "positive": [
                    "clean pure colors",
                    "fresh bright colors",
                    "harmonious colors",
                    "vibrant green",
                    "clear sky blue",
                    "clean color palette",
                    "pleasing color combination",
                    "soft natural tones"
                ],
                "weight": 0.35
            },
            "beauty": {
                "positive": [
                    "beautiful photo",
                    "stunning image",
                    "visually pleasing",
                    "aesthetically attractive",
                    "gorgeous shot",
                    "breathtaking scenery",
                    "peaceful landscape",
                    "pleasing view",
                    "soft natural beauty"
                ],
                "weight": 0.20
            },
        }
        
        # ========== 负向维度（降低权重，只保留最关键的）==========
        self.negative_prompts = [
            # 破旧感
            "worn out",
            "dilapidated", 
            "run-down",
            "ugly",        
            "messy",      
            "cluttered",  
            "dirty",
            # 杂物
            "trash",
            "litter",
            "junk"
        ]
        
        # 负向权重从 0.65 降到 0.30
        self.negative_weight = 0.30
        
        # 预计算所有正向文本特征
        self._all_positive_prompts = []
        for dim in self.dimensions.values():
            self._all_positive_prompts.extend(dim["positive"])
        print(f"[纯美学版] 预计算 {len(self._all_positive_prompts)} 条正向文本特征...")
        self._cached_positive_features = self.clip.encode_text(self._all_positive_prompts)
        
        # 预计算负向文本特征
        print(f"[纯美学版] 预计算 {len(self.negative_prompts)} 条负向文本特征...")
        self._cached_negative_features = self.clip.encode_text(self.negative_prompts)
        
        # 记录每个维度的 prompt 索引范围
        self._dim_ranges = {}
        idx = 0
        for dim_name, dim in self.dimensions.items():
            n = len(dim["positive"])
            self._dim_ranges[dim_name] = (idx, idx + n)
            idx += n
        
        # 不同模式的权重配置
        self.modes = {
            "balanced": {
                "name": "⚖️ 平衡模式",
                "desc": "光线优先，柔和美学",
                "weights": {
                    "lighting": 0.45,
                    "color": 0.35,
                    "beauty": 0.20
                },
            },
            "portrait": {
                "name": "👤 人像优先",
                "desc": "柔和光线，肤色优先",
                "weights": {
                    "lighting": 0.45,
                    "color": 0.30,
                    "beauty": 0.25
                },
            },
            "landscape": {
                "name": "🏔️ 风景优先",
                "desc": "丰富色彩，壮丽光线",
                "weights": {
                    "lighting": 0.40,
                    "color": 0.40,
                    "beauty": 0.20
                },
            },
            "vibrant": {
                "name": "🎨 鲜艳色彩",
                "desc": "饱和色彩，强烈对比",
                "weights": {
                    "lighting": 0.30,
                    "color": 0.50,
                    "beauty": 0.20
                },
            },
            "soft": {
                "name": "🌙 柔和氛围",
                "desc": "柔和光线，淡雅色调",
                "weights": {
                    "lighting": 0.50,
                    "color": 0.30,
                    "beauty": 0.20
                },
            },
        }
        
        self._current_mode = mode
        self._update_weights()
        
        print("[纯美学版] 评分器初始化完成（删除清晰度，负向权重0.30）")
        self.show_current_mode()
    
    def _update_weights(self):
        if self._current_mode not in self.modes:
            self._current_mode = "balanced"
        self.current_mode_info = self.modes[self._current_mode]
        self.weights = self.current_mode_info["weights"]
    
    def set_mode(self, mode: str):
        if mode not in self.modes:
            print(f"  ❌ 未知模式: {mode}")
            print(f"  ✅ 可用模式: {list(self.modes.keys())}")
            return False
        self._current_mode = mode
        self._update_weights()
        self.show_current_mode()
        return True
    
    def get_current_mode(self) -> str:
        return self._current_mode
    
    def get_mode_info(self, mode: str = None) -> Dict:
        if mode is None:
            mode = self._current_mode
        return self.modes.get(mode, None)
    
    def list_modes(self) -> Dict:
        return {k: {"name": v["name"], "desc": v["desc"]} for k, v in self.modes.items()}
    
    def show_current_mode(self):
        info = self.current_mode_info
        print(f"  📌 当前模式: {info['name']}")
        print(f"  📝 模式说明: {info['desc']}")
        print(f"  ⚙️  权重: lighting={self.weights['lighting']}, color={self.weights['color']}, beauty={self.weights['beauty']}")
        print(f"  🔻 负向权重: {self.negative_weight}")
        print(f"  📝 负向词: {self.negative_prompts[:5]}...")
    
    def _to_pil_image(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
    
    def _get_batch_image_features(self, images: List) -> torch.Tensor:
        batch_features = []
        for img in images:
            feat = self.clip._get_image_features(img)
            batch_features.append(feat.cpu())
        if batch_features:
            return torch.cat(batch_features, dim=0)
        return torch.tensor([])
    
    def score(self, image, mode: Optional[str] = None) -> float:
        if mode is not None:
            self.set_mode(mode)
        
        image_features = self.clip._get_image_features(image)
        
        pos_similarities = (image_features @ self._cached_positive_features.T).squeeze(0).cpu().numpy()
        pos_total = 0
        for dim_name, weight in self.weights.items():
            if dim_name not in self._dim_ranges:
                continue
            start, end = self._dim_ranges[dim_name]
            dim_score = np.mean(pos_similarities[start:end])
            pos_total += weight * dim_score
        
        neg_similarities = (image_features @ self._cached_negative_features.T).squeeze(0).cpu().numpy()
        neg_score = np.mean(neg_similarities)
        
        total = pos_total - self.negative_weight * neg_score
        return float(total)
    
    def score_batch(self, images: List) -> List[float]:
        if not images:
            return []
        
        image_features = self._get_batch_image_features(images)
        
        pos_similarities_batch = (image_features @ self._cached_positive_features.T).cpu().numpy()
        neg_similarities_batch = (image_features @ self._cached_negative_features.T).cpu().numpy()
        
        scores = []
        for i in range(len(images)):
            pos_sims = pos_similarities_batch[i]
            pos_total = 0
            for dim_name, weight in self.weights.items():
                if dim_name not in self._dim_ranges:
                    continue
                start, end = self._dim_ranges[dim_name]
                dim_score = np.mean(pos_sims[start:end])
                pos_total += weight * dim_score
            
            neg_sims = neg_similarities_batch[i]
            neg_score = np.mean(neg_sims)
            
            total = pos_total - self.negative_weight * neg_score
            scores.append(float(total))
        
        return scores


_default_scorer = None


def get_scorer(mode: str = "balanced"):
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = CLIPScorer(mode=mode)
    return _default_scorer


def set_global_mode(mode: str):
    return get_scorer().set_mode(mode)


def get_current_mode() -> str:
    return get_scorer().get_current_mode()


def list_available_modes() -> Dict:
    return get_scorer().list_modes()


def compute_clip_aesthetic_score(image, mode: Optional[str] = None) -> float:
    return get_scorer().score(image, mode)


if __name__ == "__main__":
    scorer = get_scorer()
    print("\n可用模式:")
    for mode_name, info in scorer.list_modes().items():
        print(f"  {mode_name:12} : {info['name']} - {info['desc']}")
