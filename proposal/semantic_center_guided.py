from proposal.center_proposal import (
    generate_center_boxes
)


def generate_semantic_center_boxes(
    segments,
    img_w,
    img_h,
    min_score=0.8,
    min_area_ratio=0.01
):
    """
    Mask2Former中心引导候选框
    """

    image_area = img_w * img_h

    boxes = []

    for seg in segments:

        # 置信度过滤
        if seg["score"] < min_score:
            continue

        # 面积过滤
        if seg["area"] < image_area * min_area_ratio:
            continue

        x1, y1, x2, y2 = seg["bbox"]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        boxes.extend(
            generate_center_boxes(
                cx,
                cy,
                img_w,
                img_h
            )
        )

    return boxes