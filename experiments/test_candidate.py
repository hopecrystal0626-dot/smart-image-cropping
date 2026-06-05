import sys

print(sys.path)

import cv2

from crop.candidate_generator import generate_candidates

img = cv2.imread(
    r"data/testA/A05.jpg"
)

h, w = img.shape[:2]

boxes = generate_candidates(
    w,
    h
)

print("候选框数量:", len(boxes))