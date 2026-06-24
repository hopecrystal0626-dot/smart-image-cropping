# models/depth_detector.py
import torch
from transformers import pipeline
from PIL import Image
import numpy as np

class DepthDetector:
    def __init__(self):
        device_idx = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device_idx
        )

    def predict(self, image_rgb):
        pil_img = Image.fromarray(image_rgb)
        result = self.pipe(pil_img)
        return np.array(result["depth"])