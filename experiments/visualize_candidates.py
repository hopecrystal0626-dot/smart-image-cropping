import random
import cv2

from crop.candidate_generator import generate_candidates


# 读取图片
img = cv2.imread(r"data/testA/A07.jpg")

if img is None:
    raise ValueError("图片读取失败")

h, w = img.shape[:2]

# 生成候选框
boxes = generate_candidates(
    img_w=w,
    img_h=h
)

print("总候选框数量:", len(boxes))

# 随机抽取20个
sample_boxes = random.sample(
    boxes,
    min(40, len(boxes))
)

# 复制图片
canvas = img.copy()

# 绘制随机候选框
for i, box in enumerate(sample_boxes):

    color = (
        random.randint(0,255),
        random.randint(0,255),
        random.randint(0,255)
    )

    cv2.rectangle(
        canvas,
        (box.x1, box.y1),
        (box.x2, box.y2),
        color,
        2
    )

    cv2.putText(
        canvas,
        str(i),
        (box.x1, box.y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# 保存结果
save_path = "data/output/random_candidates.jpg"

cv2.imwrite(
    save_path,
    canvas
)

print("保存成功:", save_path)

# 显示图片
cv2.imshow(
    "Random Candidates",
    canvas
)

cv2.waitKey(0)
cv2.destroyAllWindows()