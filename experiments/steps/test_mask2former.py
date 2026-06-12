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
import numpy as np
import matplotlib.pyplot as plt

from composition.panoptic_detector import PanopticDetector


# ==================================
# 配置
# ==================================

IMG_PATH = r"data/testA/A14.jpg"

OUTPUT_DIR = r"data/output/test_mask2former"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==================================
# 读取图片
# ==================================

img_bgr = cv2.imread(IMG_PATH)

if img_bgr is None:
    raise ValueError(f"无法读取图片: {IMG_PATH}")

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

# ==================================
# Mask2Former
# ==================================

detector = PanopticDetector()

result = detector.predict(img_rgb)

scene_vector = detector.build_scene_vector(result)

# ==================================
# Scene Vector 输出
# ==================================

print("\n===== Scene Vector =====")

sorted_items = sorted(
    scene_vector.items(),
    key=lambda x: x[1],
    reverse=True
)

for k, v in sorted_items:
    print(f"{k:25s}: {v:.3f}")

# 保存文本

with open(
    os.path.join(
        OUTPUT_DIR,
        "scene_vectorA14.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    for k, v in sorted_items:
        f.write(f"{k:25s}: {v:.4f}\n")

# ==================================
# segmentation map
# ==================================

seg_map = result["segmentation"].cpu().numpy()

# ==================================
# 彩色mask
# ==================================

np.random.seed(42)

colors = np.random.randint(
    0,
    255,
    (seg_map.max() + 1, 3),
    dtype=np.uint8
)

color_mask = colors[seg_map]

# ==================================
# 保存彩色分割图
# ==================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "panoptic_colorA14.png"
    ),
    cv2.cvtColor(
        color_mask,
        cv2.COLOR_RGB2BGR
    )
)

# ==================================
# Overlay
# ==================================

overlay = (
    0.6 * img_rgb +
    0.4 * color_mask
).astype(np.uint8)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "panoptic_overlayA14.png"
    ),
    cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )
)

print("\n保存完成:")
print(OUTPUT_DIR)

print("\n当前工作目录:")
print(os.getcwd())

print("\n输出目录:")
print(os.path.abspath(OUTPUT_DIR))
print(result["segments_info"][:5])

segments = detector.get_segment_bboxes(result)
bbox_img = img_rgb.copy()
for seg in segments:

    label = seg["label"]

    x1,y1,x2,y2 = seg["bbox"]

    cv2.rectangle(
        bbox_img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    cv2.putText(
        bbox_img,
        label,
        (x1,max(20,y1-5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,0,0),
        2
    )
cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "segment_bbox.png"
    ),
    cv2.cvtColor(
        bbox_img,
        cv2.COLOR_RGB2BGR
    )
)
