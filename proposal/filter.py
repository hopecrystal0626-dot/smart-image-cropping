import numpy as np
from crop.bbox_utils import compute_iou


def nms(boxes, iou_thresh=0.6):
    """
    去重：保留高质量、不重复的box
    """

    if len(boxes) == 0:
        return []

    # 按面积排序（大框优先）
    boxes = sorted(boxes, key=lambda b: b.area, reverse=True)

    keep = []

    for box in boxes:
        should_keep = True

        for kept in keep:
            if compute_iou(box, kept) > iou_thresh:
                should_keep = False
                break

        if should_keep:
            keep.append(box)

    return keep

def coarse_filter(
    boxes,
    features_dicts,
    saliency_thresh=0.15,
    semantic_thresh=0.002,
    min_area_ratio=0.02
):
    """
    boxes + features → 过滤
    """

    filtered_boxes = []
    filtered_features = []

    for box, feat in zip(boxes, features_dicts):

        area_ratio = box.area / (feat["img_area"])

        # 1️⃣ 太小的不要
        if area_ratio < min_area_ratio:
            continue

        # 2️⃣ 没显著性不要
        if feat["saliency_mean"] < saliency_thresh:
            continue

        # 3️⃣ 没语义结构不要
        if feat["semantic_density"] < semantic_thresh:
            continue

        filtered_boxes.append(box)
        filtered_features.append(feat)

    return filtered_boxes, filtered_features