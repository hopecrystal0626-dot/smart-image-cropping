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

from composition.panoptic_detector import PanopticDetector


# ======================================
# 配置
# ======================================

IMAGE_DIR = r"data/testA"

OUTPUT_DIR = r"data/output/test_mask2former"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ======================================
# 加载模型（只加载一次）
# ======================================

detector = PanopticDetector()

# ======================================
# 只处理 A01.jpg ~ A20.jpg
# ======================================

for i in range(1, 21):

    img_name = f"A{i:02d}.jpg"

    img_path = os.path.join(
        IMAGE_DIR,
        img_name
    )

    if not os.path.exists(img_path):

        print(f"跳过: {img_name}")
        continue

    print("\n" + "=" * 60)
    print("Processing:", img_name)

    save_dir = os.path.join(
        OUTPUT_DIR,
        f"A{i:02d}"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ======================================
    # 读取图片
    # ======================================

    img_bgr = cv2.imread(img_path)

    if img_bgr is None:

        print("读取失败")
        continue

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    # ======================================
    # 推理
    # ======================================

    result = detector.predict(img_rgb)

    scene_vector = detector.build_scene_vector(
        result
    )

    # ======================================
    # 保存 scene vector
    # ======================================

    sorted_items = sorted(
        scene_vector.items(),
        key=lambda x: x[1],
        reverse=True
    )

    with open(
        os.path.join(
            save_dir,
            "scene_vector.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        for k, v in sorted_items:

            f.write(
                f"{k:25s}: {v:.4f}\n"
            )

    # ======================================
    # segmentation map
    # ======================================

    seg_map = result[
        "segmentation"
    ].cpu().numpy()

    # ======================================
    # 彩色mask
    # ======================================

    np.random.seed(42)

    colors = np.random.randint(
        0,
        255,
        (seg_map.max() + 1, 3),
        dtype=np.uint8
    )

    color_mask = colors[seg_map]

    # ======================================
    # overlay
    # ======================================

    overlay = (
        0.6 * img_rgb +
        0.4 * color_mask
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            save_dir,
            "panoptic_overlay.png"
        ),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    # ======================================
    # bbox 可视化
    # ======================================

    segments = detector.get_segment_bboxes(
        result
    )

    bbox_img = img_rgb.copy()

    IGNORE_LABELS = {

        "floor-other-merged",
        "sky-other-merged",
        "wall-other-merged",
        "ceiling-merged"

    }

    for seg in segments:

        label = seg["label"]

        if label in IGNORE_LABELS:
            continue

        x1, y1, x2, y2 = seg["bbox"]

        cv2.rectangle(
            bbox_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            bbox_img,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imwrite(
        os.path.join(
            save_dir,
            "segment_bbox.png"
        ),
        cv2.cvtColor(
            bbox_img,
            cv2.COLOR_RGB2BGR
        )
    )

    print("完成:", img_name)

print("\n全部处理完成")