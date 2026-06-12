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

from saliency.isnet_detector import ISNetDetector


IMG_PATH = r"data/testA/A01.jpg"

OUTPUT_DIR = r"data/output/test_isnet"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

img_bgr = cv2.imread(IMG_PATH)

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

detector = ISNetDetector()

mask = detector.predict(img_rgb)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "isnet_mask.png"
    ),
    mask
)

print("saved")