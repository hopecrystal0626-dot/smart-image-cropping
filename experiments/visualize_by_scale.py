import cv2
import random
from crop.candidate_generator import generate_candidates

# ======================
# 1. 读取图片
# ======================
img_path = r"data/testA/A15.jpg"
img = cv2.imread(img_path)

if img is None:
    raise ValueError("图片读取失败")

h, w = img.shape[:2]

# ======================
# 2. 生成候选框
# ======================
boxes = generate_candidates(img_w=w, img_h=h)

print("总候选框数量:", len(boxes))

# ======================
# 3. 只筛选 scale = 0.3
# ======================
target_scale = 0.5

filtered_boxes = [
    box for box in boxes
    if abs(box.scale - target_scale) < 1e-6
]

print("scale=0.3 的候选框数量:", len(filtered_boxes))

# ======================
# 4. 给每个候选框分配不同颜色
# ======================
random.seed(42)

canvas = img.copy()

for i, box in enumerate(filtered_boxes):

    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    cv2.rectangle(
        canvas,
        (box.x1, box.y1),
        (box.x2, box.y2),
        color,
        2
    )

# ======================
# 5. 标注
# ======================
cv2.putText(
    canvas,
    f"scale = {target_scale}, boxes = {len(filtered_boxes)}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

# ======================
# 6. 保存 + 显示
# ======================
save_path = f"data/output/scale_15_visual.jpg"
cv2.imwrite(save_path, canvas)

print(f"[OK] saved -> {save_path}")

cv2.imshow("scale 0.3", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()