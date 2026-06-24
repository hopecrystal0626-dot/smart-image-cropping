import numpy as np


def extract_saliency_features(
    box,
    saliency_map
):
    """
    从显著图提取特征
    """

    crop = saliency_map[
        box.y1:box.y2,
        box.x1:box.x2
    ]

    if crop.size == 0:
        return {
            "saliency_mean": 0.0,
            "saliency_sum": 0.0,
            "saliency_max": 0.0,
            "saliency_ratio": 0.0
        }

    saliency_mean = float(crop.mean())

    saliency_sum = float(crop.sum())

    saliency_max = float(crop.max())

    # 动态阈值（全图前20%显著区域）
    threshold = np.percentile(
        saliency_map,
        80
    )

    saliency_ratio = float(
        np.mean(crop >= threshold)
    )

    return {
        "saliency_mean": saliency_mean,
        "saliency_sum": saliency_sum,
        "saliency_max": saliency_max,
        "saliency_ratio": saliency_ratio
    }