# -*- coding: utf-8 -*-

"""图像调节、色温、比例约束等处理函数"""

import cv2
import numpy as np


def adjust_temperature(img, v):
    """色温调节"""
    if v == 0:
        return img
    result = img.astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] + v * 0.3, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] - v * 0.3, 0, 255)
    return result.astype(np.uint8)


def apply_adjustments(img_bgr, brightness=0, contrast=0, sharpness=0,
                      highlights=0, shadows=0, temperature=0, saturation=0,
                      exposure=0):
    """综合图像调节（整合色温）"""
    if img_bgr is None:
        return None
    if all(v == 0 for v in [brightness, contrast, sharpness, highlights, shadows, temperature, saturation, exposure]):
        return img_bgr.copy()

    img = img_bgr.astype(np.float32) / 255.0

    if exposure != 0:
        gain = 1.0 + exposure / 100.0
        img = np.clip(img * gain, 0, 1)
    img = np.clip(img + brightness / 200.0, 0, 1)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = np.clip((img - mean) * (1.0 + contrast / 200.0) + mean, 0, 1)
    if sharpness != 0:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_sharp = cv2.filter2D(img, -1, kernel)
        img = np.clip(img + (img_sharp - img) * (sharpness / 100.0) * 0.3, 0, 1)
    if highlights != 0:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(gray, (15, 15), 0)[:, :, np.newaxis]
        img = np.clip(img + (highlights / 100.0) * 0.3 * (mask ** 2), 0, 1)
    if shadows != 0:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(gray, (15, 15), 0)[:, :, np.newaxis]
        img = np.clip(img - (shadows / 100.0) * 0.3 * ((1 - mask) ** 2), 0, 1)
    if temperature != 0:
        img_uint8 = (img * 255).astype(np.uint8)
        img_uint8 = adjust_temperature(img_uint8, temperature)
        img = img_uint8.astype(np.float32) / 255.0
    if saturation != 0:
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        s_factor = 1.0 + saturation / 100.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    return (np.clip(img * 255, 0, 255)).astype(np.uint8)


def apply_aspect_ratio(bbox, img_w, img_h, ratio_w, ratio_h):
    """比例约束"""
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    target_ratio = ratio_w / ratio_h
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_h = w / target_ratio
        new_w = w
    else:
        new_w = h * target_ratio
        new_h = h
    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)
    return {'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2}