import numpy as np


def _parse_bbox(seg):
    """
    兼容两种情况：
    1. dict: {"bbox": ...}
    2. array/tuple/list: [x1,y1,x2,y2,...]
    """

    # -------------------------
    # dict情况（如果未来你改回来了）
    # -------------------------
    if isinstance(seg, dict):
        bbox = seg["bbox"]
    else:
        bbox = seg  # ❗关键：直接就是seg本身

    bbox = np.array(bbox).reshape(-1)

    if len(bbox) < 4:
        raise ValueError(f"Invalid bbox format: {bbox}")

    return (
        int(bbox[0]),
        int(bbox[1]),
        int(bbox[2]),
        int(bbox[3]),
    )


def extract_semantic_features(
    box,
    segments,
    seg_map
):

    box_area = max(box.area, 1)

    object_count = 0
    overlap_ratios = []
    scores = []
    labels = set()

    # ======================
    # bbox-level stats
    # ======================
    for seg in segments:

        sx1, sy1, sx2, sy2 = _parse_bbox(seg)

        inter_x1 = max(box.x1, sx1)
        inter_y1 = max(box.y1, sy1)
        inter_x2 = min(box.x2, sx2)
        inter_y2 = min(box.y2, sy2)

        w = max(0, inter_x2 - inter_x1)
        h = max(0, inter_y2 - inter_y1)

        overlap = w * h

        if overlap <= 0:
            continue

        object_count += 1
        overlap_ratios.append(overlap / box_area)

        # 如果 seg 是 array，没有 score/label，就跳过
        if isinstance(seg, dict):
            scores.append(seg.get("score", 0.0))
            labels.add(seg.get("label", "unknown"))

    # ======================
    # pixel-level stats
    # ======================
    crop_seg = seg_map[
        box.y1:box.y2,
        box.x1:box.x2
    ]

    unique_segments = np.unique(crop_seg)

    features = {}

    features["object_count"] = object_count

    features["semantic_density"] = (
        object_count * 10000 / box_area
    )

    features["avg_segment_score"] = (
        float(np.mean(scores)) if scores else 0.0
    )

    features["largest_object_ratio"] = (
        max(overlap_ratios) if overlap_ratios else 0.0
    )

    features["scene_diversity"] = int(len(unique_segments))

    return features