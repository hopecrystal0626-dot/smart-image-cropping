# models/depth_detector.py
import os
import sys
import torch
from transformers import pipeline
from PIL import Image
import numpy as np


def _get_model_base() -> str:
    """
    返回 weights/ 目录的绝对路径。
    - 打包后（PyInstaller）：从 sys._MEIPASS 下找
    - 正常运行：从本文件向上两级找 weights/
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller 解压目录
        return os.path.join(sys._MEIPASS, "weights")
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")


class DepthDetector:
    def __init__(self):
        device_idx = 0 if torch.cuda.is_available() else -1

        local_path = os.path.join(_get_model_base(), "depth-anything-v2-small")

        if os.path.isdir(local_path):
            # 优先使用本地离线模型
            model_src = local_path
            print(f"[DepthDetector] 使用本地模型：{local_path}")
        else:
            # 回退到在线下载（需要网络）
            model_src = "depth-anything/Depth-Anything-V2-Small-hf"
            print("[DepthDetector] 本地模型不存在，尝试在线下载...")

        self.pipe = pipeline(
            task="depth-estimation",
            model=model_src,
            device=device_idx,
        )

    def predict(self, image_rgb):
        pil_img = Image.fromarray(image_rgb)
        result = self.pipe(pil_img)
        return np.array(result["depth"])