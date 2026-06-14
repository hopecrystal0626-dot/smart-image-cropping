from features.saliency_feature import extract_saliency_features
from features.semantic_feature import extract_semantic_features
from features.depth_feature import extract_depth_features

import numpy as np


# =========================
# Step 1: 统一归一化
# =========================



# =========================
# Feature Extractor
# =========================
def extract_all_features(
    box,
    saliency_map,
    seg_map,
    segments,
    depth_map
):

    features = {}

    # saliency
    features.update(
        extract_saliency_features(
            box,
            saliency_map
        )
    )

    # semantic
    features.update(
        extract_semantic_features(
            box,
            segments,
            seg_map
        )
    )

    # depth
    features.update(
        extract_depth_features(
            box,
            depth_map
        )
    )

    

    return features