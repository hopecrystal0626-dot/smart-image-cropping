import numpy as np


def extract_depth_features(
    box,
    depth_map
):

    crop = depth_map[
        box.y1:box.y2,
        box.x1:box.x2
    ]

    if crop.size == 0:
        return {
            "depth_mean": 0.0,
            "depth_std": 0.0,
            "depth_min": 0.0,
            "depth_max": 0.0
        }

    return {
        "depth_mean": float(crop.mean()),
        "depth_std": float(crop.std()),
        "depth_min": float(crop.min()),
        "depth_max": float(crop.max())
    }