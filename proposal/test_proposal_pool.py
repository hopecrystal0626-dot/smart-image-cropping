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

from composition.panoptic_detector import (
    PanopticDetector
)

from proposal.proposal_generator import (
    generate_all_proposals
)

IMG_PATH = r"data/testA/A01.jpg"

img_bgr = cv2.imread(IMG_PATH)

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

# ------------------------
# U2Net
# ------------------------

saliency_detector = U2NetDetector()

saliency_map = saliency_detector.predict(
    img_rgb
)

# ------------------------
# Mask2Former
# ------------------------

panoptic_detector = PanopticDetector()

result = panoptic_detector.predict(
    img_rgb
)

segments = (
    panoptic_detector.get_segment_bboxes(
        result
    )
)

# ------------------------
# Proposal Pool
# ------------------------

all_boxes = generate_all_proposals(
    img_rgb,
    saliency_map,
    segments
)

print()

print(
    "最终候选框数量:",
    len(all_boxes)
)