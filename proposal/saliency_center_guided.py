import numpy as np

from proposal.center_proposal import (
    generate_center_boxes
)


def generate_saliency_center_boxes(
    saliency_map,
    img_w,
    img_h,
    threshold=128
):
    """
    U2Net中心引导
    """

    mask = saliency_map > threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return []

    cx = xs.mean()
    cy = ys.mean()

    return generate_center_boxes(
        cx,
        cy,
        img_w,
        img_h
    )