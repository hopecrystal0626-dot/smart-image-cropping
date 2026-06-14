import cv2
import numpy as np

from crop.bbox_utils import BBox


def adjust_to_aspect_ratio(
    x1,
    y1,
    x2,
    y2,
    target_ratio,
    img_w,
    img_h
):
    """
    调整bbox到指定宽高比
    """

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    current_ratio = bw / bh

    if current_ratio > target_ratio:

        new_h = bw / target_ratio
        new_w = bw

    else:

        new_w = bh * target_ratio
        new_h = bh

    nx1 = int(cx - new_w / 2)
    nx2 = int(cx + new_w / 2)

    ny1 = int(cy - new_h / 2)
    ny2 = int(cy + new_h / 2)

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)

    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def expand_bbox(
    x1,
    y1,
    x2,
    y2,
    scale,
    img_w,
    img_h
):
    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = bw * scale
    new_h = bh * scale

    nx1 = int(cx - new_w / 2)
    nx2 = int(cx + new_w / 2)

    ny1 = int(cy - new_h / 2)
    ny2 = int(cy + new_h / 2)

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)

    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def generate_saliency_guided_boxes(
    saliency_map,
    img_w,
    img_h,
    threshold=128,
    min_area_ratio=0.01,
    expand_scales=(1.0, 1.2, 1.5, 2.0)
):
    """
    U2Net显著性引导框
    """

    target_ratio = img_w / img_h

    binary = (
        saliency_map > threshold
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    boxes = []

    min_area = (
        img_w *
        img_h *
        min_area_ratio
    )

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        x1 = x
        y1 = y

        x2 = x + w
        y2 = y + h

        for scale in expand_scales:

            ex1, ey1, ex2, ey2 = (
                expand_bbox(
                    x1,
                    y1,
                    x2,
                    y2,
                    scale,
                    img_w,
                    img_h
                )
            )

            ex1, ey1, ex2, ey2 = (
                adjust_to_aspect_ratio(
                    ex1,
                    ey1,
                    ex2,
                    ey2,
                    target_ratio,
                    img_w,
                    img_h
                )
            )

            boxes.append(
                BBox(
                    ex1,
                    ey1,
                    ex2,
                    ey2,
                    scale
                )
            )

    return boxes