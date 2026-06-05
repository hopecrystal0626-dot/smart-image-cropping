import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composition.thirds import compute_thirds_score
import cv2
import numpy as np

# 创建一张假图片
img = np.zeros((480, 640, 3), dtype=np.uint8)

# 测试不同位置的框
boxes = [
    (0, 0, 320, 240),      # 左上角
    (160, 120, 320, 240),  # 中心
    (320, 240, 320, 240),  # 右下角
]

print("三分法测试:")
for box in boxes:
    score = compute_thirds_score(img, box)
    print(f"框 {box} → 三分法得分: {score:.4f}")