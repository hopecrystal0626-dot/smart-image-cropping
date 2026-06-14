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

from saliency.u2net_detector import (
    U2NetDetector
)

from proposal.saliency_guided import (
    generate_saliency_guided_boxes
)

IMG_PATH = r"data/testA/A07.jpg"

img_bgr = cv2.imread(
    IMG_PATH
)

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

h, w = img_rgb.shape[:2]

# ==========================
# U2Net
# ==========================

detector = U2NetDetector()

saliency_map = detector.predict(
    img_rgb
)

print(
    "saliency:",
    saliency_map.min(),
    saliency_map.max(),
    saliency_map.mean()
)

# ==========================
# Generate Boxes
# ==========================

boxes = generate_saliency_guided_boxes(
    saliency_map,
    w,
    h,
    threshold=80
)

print(
    "Saliency Boxes:",
    len(boxes)
)

# ==========================
# Visualize
# ==========================

vis = img_bgr.copy()

for box in boxes:

    cv2.rectangle(
        vis,
        (box.x1, box.y1),
        (box.x2, box.y2),
        (0, 255, 0),
        2
    )

cv2.imwrite(
    "saliency_guided_boxes.jpg",
    vis
)

print(
    "保存完成:",
    "saliency_guided_boxes.jpg"
)