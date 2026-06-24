# proposal/center_proposal.py

from crop.bbox_utils import BBox


def generate_center_boxes(
    center_x,
    center_y,
    img_w,
    img_h,
    scales=(0.20, 0.30, 0.40, 0.50, 0.60),
    use_offsets=True
):
    """
    根据中心点生成候选框

    返回:
        List[BBox]
    """

    boxes = []

    for scale in scales:

        crop_w = int(img_w * scale)
        crop_h = int(img_h * scale)

        # =====================
        # 中心偏移
        # =====================

        if use_offsets:

            offsets = [
                (0.00, 0.00),   # 原中心

                (-0.15, 0.00),  # 左
                ( 0.15, 0.00),  # 右

                (0.00, -0.10),  # 上
                (0.00,  0.10),  # 下
            ]

        else:

            offsets = [
                (0.00, 0.00)
            ]

        # =====================
        # 为每个偏移生成框
        # =====================

        for ox, oy in offsets:

            cx = center_x + ox * crop_w
            cy = center_y + oy * crop_h

            x1 = int(cx - crop_w / 2)
            y1 = int(cy - crop_h / 2)

            x2 = x1 + crop_w
            y2 = y1 + crop_h

            # =====================
            # 越界修正
            # =====================

            if x1 < 0:
                x2 -= x1
                x1 = 0

            if y1 < 0:
                y2 -= y1
                y1 = 0

            if x2 > img_w:
                shift = x2 - img_w
                x1 -= shift
                x2 = img_w

            if y2 > img_h:
                shift = y2 - img_h
                y1 -= shift
                y2 = img_h

            x1 = max(0, x1)
            y1 = max(0, y1)

            boxes.append(
                BBox(
                    x1,
                    y1,
                    x2,
                    y2,
                    scale
                )
            )

    return boxes