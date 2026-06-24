# core/inference.py
import cv2
import numpy as np
import torch
from PIL import Image
from smart_framing.models.u2net_detector import U2NetDetector
from smart_framing.models.panoptic_detector import PanopticDetector
from smart_framing.models.depth_detector import DepthDetector
from smart_framing.models.aesthetic_detector import AestheticDetector
from smart_framing import config

# ---------- 全局单例 ----------
_u2net = None
_panoptic = None
_depth = None
_aesthetic = None

def get_u2net():
    global _u2net
    if _u2net is None:
        _u2net = U2NetDetector(model_path=config.U2NET_MODEL_PATH)
    return _u2net

def get_panoptic():
    global _panoptic
    if _panoptic is None:
        _panoptic = PanopticDetector(model_name=config.PANOPTIC_MODEL_NAME)
    return _panoptic

def get_depth():
    global _depth
    if _depth is None:
        _depth = DepthDetector()   # 内部使用 config.DEPTH_MODEL_NAME
    return _depth

def get_aesthetic():
    global _aesthetic
    if _aesthetic is None:
        _aesthetic = AestheticDetector(model_path=config.AESTHETIC_MODEL_PATH)
    return _aesthetic

# ---------- 对外推理函数 ----------

def compute_saliency(img_rgb):
    """返回归一化显著性掩码 (0~1 float32, HxW)"""
    detector = get_u2net()
    mask = detector.predict(img_rgb)          # uint8 0~255
    mask = mask.astype(np.float32) / 255.0
    return mask

def compute_panoptic(img_rgb):
    """返回 (result, segments, seg_map)"""
    detector = get_panoptic()
    result = detector.predict(img_rgb)
    segments = detector.get_segment_bboxes(result)
    seg_map = result["segmentation"].cpu().numpy()
    return result, segments, seg_map

def compute_depth(img_rgb):
    """返回归一化深度图 (0~1 float32, HxW)"""
    detector = get_depth()
    depth_raw = detector.predict(img_rgb)
    if depth_raw.shape[:2] != img_rgb.shape[:2]:
        depth_raw = cv2.resize(depth_raw, (img_rgb.shape[1], img_rgb.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max - d_min > 1e-6:
        depth = (depth_raw - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth_raw)
    return depth

def aesthetic_score(crop_rgb):
    """对裁剪图像计算美学分（直接使用 AestheticDetector 的内部模型）"""
    detector = get_aesthetic()
    pil_img = Image.fromarray(crop_rgb)
    image_tensor = detector.preprocess(pil_img).unsqueeze(0).to(detector.device)
    with torch.no_grad():
        image_features = detector.clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = detector.predictor(image_features).item()
    return score