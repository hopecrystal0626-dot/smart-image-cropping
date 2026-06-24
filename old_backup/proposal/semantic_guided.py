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
    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    current_ratio = bw / bh

    if current_ratio > target_ratio:

        new_w = bw
        new_h = bw / target_ratio

    else:

        new_h = bh
        new_w = bh * target_ratio

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

    nw = bw * scale
    nh = bh * scale

    nx1 = int(cx - nw / 2)
    nx2 = int(cx + nw / 2)

    ny1 = int(cy - nh / 2)
    ny2 = int(cy + nh / 2)

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)

    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def generate_semantic_guided_boxes(
    segments,
    img_w,
    img_h,
    min_area_ratio=0.01,
    min_score=0.80,
    expand_scales=(1.0, 1.3, 1.6, 2.0)
):
    """
    Mask2Former语义引导框
    """

    target_ratio = img_w / img_h

    image_area = img_w * img_h

    boxes = []

    for seg in segments:

        if seg["score"] < min_score:
            continue

        if seg["area"] < image_area * min_area_ratio:
            continue

        x1, y1, x2, y2 = seg["bbox"]

        for scale in expand_scales:

            ex1, ey1, ex2, ey2 = expand_bbox(
                x1,
                y1,
                x2,
                y2,
                scale,
                img_w,
                img_h
            )

            ex1, ey1, ex2, ey2 = adjust_to_aspect_ratio(
                ex1,
                ey1,
                ex2,
                ey2,
                target_ratio,
                img_w,
                img_h
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