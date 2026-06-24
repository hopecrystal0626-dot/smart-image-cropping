# composition/aesthetic_scorer.py
import os
import torch
import cv2
import numpy as np
from PIL import Image
import pyiqa

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class AestheticScorer:
    def __init__(self, device=None, model_name='nima'):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"AestheticScorer using {self.device}")
        self.model = pyiqa.create_metric(model_name, device=self.device)
        self.model.eval()

    def preprocess(self, image):
        """
        将输入图像（PIL Image 或 numpy array）转换为 [0,1] 范围的 tensor，形状 (1,3,H,W)
        """
        if isinstance(image, np.ndarray):
            # 假设是 RGB 格式，0-255 uint8
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                # 已经是 float，但可能超出范围，手动 clip
                image = np.clip(image, 0.0, 1.0)
            img_tensor = torch.from_numpy(image).float()
            # 确保形状 (H,W,C) -> (C,H,W)
            if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
        elif isinstance(image, Image.Image):
            img_np = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        else:
            raise TypeError("Unsupported image type")
        # 添加 batch 维度
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        # 确保范围 [0,1]
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        return img_tensor

    def get_score(self, image):
        """
        image: PIL Image 或 RGB numpy array (0-255 uint8)
        返回美学评分 (float)
        """
        img_tensor = self.preprocess(image)
        with torch.no_grad():
            score = self.model(img_tensor)
        if isinstance(score, tuple):
            score = score[0]
        return float(score.item())

    def get_score_from_cv2(self, image_bgr):
        """从 OpenCV BGR 图像计算评分"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.get_score(rgb)

    def get_score_for_bbox(self, image_bgr, bbox):
        """计算候选框区域的美学评分"""
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        h, w = image_bgr.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(x1+1, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(y1+1, min(y2, h))
        cropped = image_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            return 0.0
        return self.get_score_from_cv2(cropped)