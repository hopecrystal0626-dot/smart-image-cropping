"""CLIP 模型封装模块 (针对 transformers 5.10.2)"""

import torch
from PIL import Image
import numpy as np
from typing import Union, List, Optional
from transformers import CLIPModel, CLIPProcessor

import warnings
warnings.filterwarnings("ignore")


class CLIPScorer:
    """使用 HuggingFace transformers 的 CLIP 评分器"""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("[CLIP] 正在加载模型...")
        print(f"[CLIP] 使用设备: {self.device}")

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model.eval()

        print("[CLIP] 模型加载完成！")

    def _get_image_features(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """提取图像的 CLIP 特征向量"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # 确保返回的是 tensor（兼容不同 transformers 版本）
            if hasattr(image_features, 'pooler_output') and image_features.pooler_output is not None:
                image_features = image_features.pooler_output

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def compute_similarity(
        self,
        image: Union[str, np.ndarray, Image.Image],
        texts: List[str]
    ) -> np.ndarray:
        """计算图像与文本列表的相似度"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        inputs = self.processor(
            text=texts, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embeds = image_embeds / image_embeds.norm(
                dim=-1, keepdim=True
            )
            text_embeds = text_embeds / text_embeds.norm(
                dim=-1, keepdim=True
            )

            similarities = (image_embeds @ text_embeds.T).squeeze(0)

        similarities = similarities.cpu().numpy()
        return similarities

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """提取文本的 CLIP 特征向量"""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            text_features = outputs.pooler_output

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        return text_features


_CLIP_MODEL = None


def get_clip_model(device: Optional[str] = None) -> CLIPScorer:
    """获取 CLIP 模型单例"""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        _CLIP_MODEL = CLIPScorer(device=device)
    return _CLIP_MODEL
