# proposal/saliency_center_guided.py
import numpy as np
from smart_framing.proposals.center_proposal import generate_center_boxes

def generate_saliency_center_boxes(saliency_map, img_w, img_h, threshold=128):
    mask = saliency_map > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    cx = xs.mean()
    cy = ys.mean()
    return generate_center_boxes(cx, cy, img_w, img_h)