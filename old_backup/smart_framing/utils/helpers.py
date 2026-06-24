# utils/helpers.py
import cv2
import os
import numpy as np

def load_image(img_path):
    """读取图片，返回 RGB 格式"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"图片不存在: {img_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def load_framing(img_path):
    """尝试读取 _framing.jpg，若无则返回 None"""
    base, ext = os.path.splitext(img_path)
    framing_path = base + "_framing.jpg"
    if not os.path.exists(framing_path):
        return None
    img_bgr = cv2.imread(framing_path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def resolve_img_ids(img_arg, data_dir):
    """用于批处理脚本（可选），pipeline 中不使用"""
    if img_arg.lower() == "all":
        ids = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            if "_framing" in fname:
                continue
            ids.append(os.path.splitext(fname)[0])
        return ids
    return [s.strip() for s in img_arg.split(",") if s.strip()]