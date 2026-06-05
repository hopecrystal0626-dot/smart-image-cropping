import cv2
import numpy as np
import time
from typing import Tuple, Optional
from pathlib import Path

import os
# ========== 设置模型下载路径（如需使用 U²-Net）==========
#MODEL_DIR = r"D:\U2NET"
#os.environ['U2NET_HOME'] = MODEL_DIR
#os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- 基类 ----------
class SaliencyDetector:
    def __init__(self, name: str):
        self.name = name

    def detect(self, image: np.ndarray) -> np.ndarray:
        """输入BGR图像，输出float32显著图，范围[0,1]"""
        raise NotImplementedError

# ---------- 1. 谱残差 (纯NumPy实现) ----------
'''class SpectralResidualDetector(SaliencyDetector):
    def __init__(self):
        super().__init__("SpectralResidual")

    def detect(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        log_mag = np.log(magnitude + 1e-8)
        kernel = np.ones((3,3)) / 9.0
        smooth_log_mag = cv2.filter2D(log_mag, -1, kernel)
        spectral_residual = log_mag - smooth_log_mag
        sal_map = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase)))
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
        return sal_map.astype(np.float32)

# ---------- 2. HC (Histogram Contrast) 简化版 ----------
class HCDetector(SaliencyDetector):
    def __init__(self):
        super().__init__("HC")

    def detect(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        pixels = lab.reshape(-1, 3)
        if len(pixels) > 5000:
            idx = np.random.choice(len(pixels), 5000, replace=False)
            sample = pixels[idx]
        else:
            sample = pixels
        sal_sample = np.zeros(len(sample))
        for i, p in enumerate(sample):
            dist = np.linalg.norm(sample - p, axis=1)
            sal_sample[i] = np.mean(dist)
        sal_sample = (sal_sample - sal_sample.min()) / (sal_sample.max() - sal_sample.min() + 1e-8)
        if len(sample) < len(pixels):
            sal_map = np.ones(len(pixels)) * np.mean(sal_sample)
        else:
            sal_map = sal_sample
        sal_map = sal_map.reshape(h, w).astype(np.float32)
        sal_map = cv2.GaussianBlur(sal_map, (5,5), 1)
        return sal_map
'''

# ---------- 3. FT (Frequency Tuned) ----------
class FTDetector(SaliencyDetector):
    def __init__(self):
        super().__init__("FT")

    def detect(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5,5), 1)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_L = np.mean(lab[:,:,0])
        mean_a = np.mean(lab[:,:,1])
        mean_b = np.mean(lab[:,:,2])
        diff_L = lab[:,:,0] - mean_L
        diff_a = lab[:,:,1] - mean_a
        diff_b = lab[:,:,2] - mean_b
        sal_map = np.sqrt(diff_L**2 + diff_a**2 + diff_b**2)
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
        sal_map = cv2.GaussianBlur(sal_map, (3,3), 0.5)
        return sal_map.astype(np.float32)

# ---------- 4. U²-Net via rembg（无需手动定义网络结构）----------
'''try:
    from rembg import remove
    import numpy as np
    import cv2
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("警告: rembg未安装，请运行 pip install rembg")

if REMBG_AVAILABLE:
    class U2NetRembgDetector(SaliencyDetector):
        def __init__(self):
            super().__init__("U2Net_Rembg")
            # rembg 会在首次使用时自动下载模型到 ~/.u2net/u2net.onnx
        
        def detect(self, image: np.ndarray) -> np.ndarray:
            # 输入 BGR，需要转 RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # only_mask=True 返回前景掩膜 (0-255 uint8)
            mask = remove(rgb, only_mask=True)
            # 转换为 float32 并归一化到 [0,1]
            sal_map = np.array(mask, dtype=np.float32) / 255.0
            return sal_map

'''

# ========== 全局默认检测器（用于 saliency_utils）==========
_default_detector = None

def get_default_detector():
    global _default_detector
    if _default_detector is None:
        _default_detector = FTDetector()   # 默认使用 FT 方法
    return _default_detector