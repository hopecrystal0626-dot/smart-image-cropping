import sys
import os

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, ROOT)

import cv2

from saliency.u2net_detector import U2NetDetector

from composition.panoptic_detector import PanopticDetector

from proposal.saliency_center_guided import (
    generate_saliency_center_boxes
)

from proposal.semantic_center_guided import (
    generate_semantic_center_boxes
)

# =========================
# 图片
# =========================

IMG_PATH = r"data/testA/A07.jpg"

img_bgr = cv2.imread(IMG_PATH)

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

h, w = img_rgb.shape[:2]

# =========================
# U2Net
# =========================

print("Loading U2Net...")

saliency_detector = U2NetDetector()

saliency_map = saliency_detector.predict(
    img_rgb
)

saliency_boxes = (
    generate_saliency_center_boxes(
        saliency_map,
        w,
        h
    )
)

print(
    "Saliency Center Boxes:",
    len(saliency_boxes)
)

# =========================
# Mask2Former
# =========================

print("Loading Mask2Former...")

panoptic_detector = PanopticDetector()

result = panoptic_detector.predict(
    img_rgb
)

segments = (
    panoptic_detector.get_segment_bboxes(
        result
    )
)

semantic_boxes = (
    generate_semantic_center_boxes(
        segments,
        w,
        h
    )
)

print(
    "Semantic Center Boxes:",
    len(semantic_boxes)
)

# =========================
# 可视化
# =========================

vis = img_bgr.copy()

# 显著性框（绿色）

for box in saliency_boxes:

    cv2.rectangle(
        vis,
        (box.x1, box.y1),
        (box.x2, box.y2),
        (0, 255, 0),
        2
    )

# 语义框（红色）

for box in semantic_boxes:

    cv2.rectangle(
        vis,
        (box.x1, box.y1),
        (box.x2, box.y2),
        (0, 0, 255),
        2
    )

cv2.imwrite(
    "center_guided_boxes.jpg",
    vis
)

print(
    "\n保存完成：center_guided_boxes.jpg"
)