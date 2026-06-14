import sys
import os

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(0, ROOT)

import cv2

from saliency.u2net_detector import U2NetDetector


IMG_PATH = r"data/testA/A15.jpg"

OUTPUT_DIR = r"data/output/test_u2net"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

img_bgr = cv2.imread(IMG_PATH)

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

detector = U2NetDetector()

mask = detector.predict(img_rgb)
print(mask.min())
print(mask.max())
print(mask.mean())

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "u2net_mask.png"
    ),
    mask
)

print("保存完成")