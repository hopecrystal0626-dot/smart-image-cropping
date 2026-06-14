import sys
import os
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# =========================
# 路径设置
# =========================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from saliency.u2net_detector import U2NetDetector
from composition.panoptic_detector import PanopticDetector
from proposal.proposal_generator import generate_all_proposals
from features.extract_all_features import extract_all_features


# =========================
# IoU for NMS
# =========================
def iou(box1, box2):
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = box1.area
    area2 = box2.area

    return inter / (area1 + area2 - inter)


# =========================
# simple NMS
# =========================
def nms(boxes, scores, iou_thresh=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        rest = []
        for j in idxs[1:]:
            if iou(boxes[i], boxes[j]) < iou_thresh:
                rest.append(j)

        idxs = np.array(rest)

    return keep


# =========================
# 读取图片
# =========================
IMG_PATH = r"data/testA/A17.jpg"

img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise ValueError("image not found")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

print("Image:", w, "x", h)


# =========================
# 1. Saliency
# =========================
print("\n[1] Saliency...")

u2net = U2NetDetector()
saliency_map = u2net.predict(img_rgb)

print("Saliency range:", saliency_map.min(), saliency_map.max())


# =========================
# 2. Mask2Former
# =========================
print("\n[2] Mask2Former...")

panoptic = PanopticDetector()
result = panoptic.predict(img_rgb)

seg_map = result["segmentation"]
if hasattr(seg_map, "cpu"):
    seg_map = seg_map.cpu().numpy()
seg_map = np.array(seg_map)

segments = panoptic.get_segment_bboxes(result)

print("Segments:", len(segments))


# =========================
# 3. Depth
# =========================
print("\n[3] Depth...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

depth_map = np.array(
    depth_pipe(Image.fromarray(img_rgb))["depth"]
)

print("Depth ready")


# =========================
# 4. Proposals
# =========================
print("\n[4] proposals...")

boxes = generate_all_proposals(
    img_rgb,
    saliency_map,
    segments
)

print("before NMS:", len(boxes))


# =========================
# 5. crude scoring for NMS
# =========================
# 用一个简单 saliency mean 做排序
def quick_score(box):
    crop = saliency_map[box.y1:box.y2, box.x1:box.x2]
    return float(crop.mean()) if crop.size > 0 else 0


scores = [quick_score(b) for b in boxes]


# =========================
# 6. NMS 去重
# =========================
keep_idx = nms(boxes, scores, iou_thresh=0.5)
boxes = [boxes[i] for i in keep_idx]

print("after NMS:", len(boxes))


# =========================
# 7. 粗筛（saliency + size）
# =========================
filtered = []

for b in boxes:
    crop = saliency_map[b.y1:b.y2, b.x1:b.x2]
    if crop.size == 0:
        continue

    sal_mean = crop.mean()

    if sal_mean < 5:   # 🔥粗筛阈值（可调）
        continue

    if b.area < (h * w * 0.01):
        continue

    filtered.append(b)

print("after filter:", len(filtered))


# =========================
# 8. 取 Top20
# =========================
filtered = sorted(
    filtered,
    key=lambda b: quick_score(b),
    reverse=True
)[:10]


print("\nFinal Top20:", len(filtered))


# =========================
# 9. 可视化
# =========================
vis = img_bgr.copy()

for i, b in enumerate(filtered):
    cv2.rectangle(
        vis,
        (b.x1, b.y1),
        (b.x2, b.y2),
        (0, 255, 0),
        2
    )
    cv2.putText(
        vis,
        str(i),
        (b.x1, b.y1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

cv2.imwrite("top10_boxes.jpg", vis)

print("\nSaved: top10_boxes.jpg")