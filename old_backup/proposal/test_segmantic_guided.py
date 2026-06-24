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

from composition.panoptic_detector import (
    PanopticDetector
)

from proposal.semantic_guided import (
    generate_semantic_guided_boxes
)

IMG_PATH = r"data/testA/A17.jpg"

img_bgr = cv2.imread(
    IMG_PATH
)

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

h, w = img_rgb.shape[:2]

# --------------------------
# Mask2Former
# --------------------------

detector = PanopticDetector()

result = detector.predict(
    img_rgb
)

segments = detector.get_segment_bboxes(
    result
)

print(
    "Segments:",
    len(segments)
)

# --------------------------
# Generate
# --------------------------

boxes = generate_semantic_guided_boxes(
    segments,
    w,
    h
)

print(
    "Semantic Boxes:",
    len(boxes)
)

# --------------------------
# Visualize
# --------------------------

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
    "semantic_guided_boxes.jpg",
    vis
)

print(
    "保存完成: semantic_guided_boxes.jpg"
)