# models/panoptic_detector.py
import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


def _get_model_base() -> str:
    """
    返回 weights/ 目录的绝对路径。
    - 打包后（PyInstaller）：从 sys._MEIPASS 下找
    - 正常运行：从本文件向上两级找 weights/
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "weights")
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")


class PanopticDetector:
    def __init__(self, model_name="facebook/mask2former-swin-base-coco-panoptic"):
        print("Loading Mask2Former...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        local_path = os.path.join(_get_model_base(), "mask2former-swin-base-coco-panoptic")

        if os.path.isdir(local_path):
            model_src = local_path
            print(f"[PanopticDetector] 使用本地模型：{local_path}")
        else:
            model_src = model_name
            print("[PanopticDetector] 本地模型不存在，尝试在线下载...")

        self.processor = AutoImageProcessor.from_pretrained(model_src)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_src)
        self.model.to(self.device)
        self.model.eval()
        print("Mask2Former loaded.")

    def predict(self, image_rgb):
        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])],
        )[0]
        return result

    def build_scene_vector(self, result):
        seg_map = result["segmentation"].cpu().numpy()
        h, w = seg_map.shape
        total_pixels = h * w
        scene_vector = {}
        for seg in result["segments_info"]:
            segment_id = seg["id"]
            label_id = seg["label_id"]
            label_name = self.model.config.id2label.get(label_id, str(label_id))
            area = np.sum(seg_map == segment_id)
            ratio = area / total_pixels
            scene_vector[label_name] = scene_vector.get(label_name, 0) + ratio
        return scene_vector

    def get_segment_bboxes(self, result):
        seg_map = result["segmentation"].cpu().numpy()
        segments = []
        for seg in result["segments_info"]:
            segment_id = seg["id"]
            label_id = seg["label_id"]
            label_name = self.model.config.id2label.get(label_id, str(label_id))
            ys, xs = np.where(seg_map == segment_id)
            if len(xs) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            area = len(xs)
            segments.append({
                "label": label_name,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "score": seg["score"],
            })
        return segments